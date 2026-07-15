import os
import csv
import json
import argparse
import binascii
import gc
import random
import shutil
import multiprocessing as mp
from types import SimpleNamespace

from scapy.all import IP, TCP, UDP

try:
    from process_pretrain_data.flow_sessionizer import collect_sessions_from_pcap
except ModuleNotFoundError:
    from flow_sessionizer import collect_sessions_from_pcap

_GLOBAL_LABEL_MAP = None


def init_worker(label_map):
    """
    多进程 worker 初始化函数。

    Windows 下每个子进程会复制一份 label_map。
    如果 CSV 很大，内存会增加；第一版先这样做，稳定优先。
    """
    global _GLOBAL_LABEL_MAP
    _GLOBAL_LABEL_MAP = label_map


def get_packet_5tuple(pkt):
    """
    从 packet 中提取五元组：
    src_ip, src_port, dst_ip, dst_port, proto

    当前版本只处理：
    IPv4 + TCP
    IPv4 + UDP

    注意：
    TLS / HTTPS / HTTP / SSH / DNS / QUIC 等应用层协议，
    只要它们跑在 TCP 或 UDP 上，这里都会被处理。
    """
    if IP not in pkt:
        return None

    ip = pkt[IP]
    src_ip = ip.src
    dst_ip = ip.dst
    proto = ip.proto

    if TCP in pkt:
        src_port = pkt[TCP].sport
        dst_port = pkt[TCP].dport
    elif UDP in pkt:
        src_port = pkt[UDP].sport
        dst_port = pkt[UDP].dport
    else:
        return None

    return src_ip, src_port, dst_ip, dst_port, proto


def normalize_proto(proto):
    """
    将协议字段统一成数字：
    TCP -> 6
    UDP -> 17

    CSV 中可能是：
    6
    17
    TCP
    UDP
    tcp
    udp
    """
    if proto is None:
        return None

    proto_str = str(proto).strip().upper()

    if proto_str == "TCP":
        return 6
    if proto_str == "UDP":
        return 17
    if proto_str == "ICMP":
        return 1

    try:
        return int(float(proto_str))
    except ValueError:
        return None


def make_biflow_key(src_ip, src_port, dst_ip, dst_port, proto):
    """
    构造双向 flow key。

    目的：
    A:12345 -> B:443
    B:443   -> A:12345

    这两个方向归入同一个 flow。
    """
    src_port = int(src_port)
    dst_port = int(dst_port)
    proto = normalize_proto(proto)

    endpoint_a = (src_ip, src_port)
    endpoint_b = (dst_ip, dst_port)

    if endpoint_a <= endpoint_b:
        return src_ip, src_port, dst_ip, dst_port, proto
    else:
        return dst_ip, dst_port, src_ip, src_port, proto


def get_direction(src_ip, src_port, flow_key):
    """
    判断 packet 在双向 flow 中的方向。

    dir = 0:
        flow_key 中 endpoint1 -> endpoint2

    dir = 1:
        flow_key 中 endpoint2 -> endpoint1
    """
    ip1, port1, ip2, port2, proto = flow_key

    if src_ip == ip1 and int(src_port) == int(port1):
        return 0
    else:
        return 1


def packet_to_hex_tokens(pkt, token_bytes=2, max_packet_tokens=None, cut_link_layer=False):
    """
    将 packet 转成十六进制 token。

    token_bytes=2 表示每 2 字节作为一个 token。

    例如：
    原始 hex:
        45000034abcd4000

    tokens:
        4500 0034 abcd 4000

    cut_link_layer=True：
        从 IP 层开始取 bytes，去掉以太网链路层头。

    cut_link_layer=False：
        保留完整 packet bytes。
    """
    if cut_link_layer and IP in pkt:
        raw_bytes = bytes(pkt[IP])
    else:
        raw_bytes = bytes(pkt)

    hex_str = binascii.hexlify(raw_bytes).decode("utf-8")

    step = token_bytes * 2
    tokens = []

    for i in range(0, len(hex_str), step):
        token = hex_str[i:i + step]
        if len(token) == step:
            tokens.append(token)

    if max_packet_tokens is not None:
        tokens = tokens[:max_packet_tokens]

    return tokens


def load_csv_labels(
    csv_path,
    src_ip_col,
    src_port_col,
    dst_ip_col,
    dst_port_col,
    proto_col,
    label_col,
    encoding="utf-8-sig",
):
    """
    从 CSV 中加载标签。

    返回：
    label_map = {
        biflow_key: label
    }

    注意：
    CSV 中的五元组方向可能和 pcap 中相反，
    所以这里统一转成双向 flow key。
    """
    label_map = {}

    if csv_path is None:
        raise ValueError("label_mode=csv 时必须提供 --label_csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    with open(csv_path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)

        required_cols = [
            src_ip_col,
            src_port_col,
            dst_ip_col,
            dst_port_col,
            proto_col,
            label_col,
        ]

        for col in required_cols:
            if col not in reader.fieldnames:
                raise ValueError(
                    f"CSV 中找不到列: {col}\n"
                    f"当前 CSV 表头为: {reader.fieldnames}"
                )

        total_rows = 0
        valid_rows = 0

        for row in reader:
            total_rows += 1

            try:
                src_ip = row[src_ip_col].strip()
                dst_ip = row[dst_ip_col].strip()
                src_port = int(float(row[src_port_col]))
                dst_port = int(float(row[dst_port_col]))
                proto = normalize_proto(row[proto_col])
                label = row[label_col].strip()
            except Exception:
                continue

            if proto is None:
                continue

            try:
                key = make_biflow_key(
                    src_ip,
                    src_port,
                    dst_ip,
                    dst_port,
                    proto,
                )
            except Exception:
                continue

            label_map[key] = label
            valid_rows += 1

    print(f"[CSV] loaded labels: {valid_rows}/{total_rows}")
    return label_map


def infer_label_from_path(pcap_path):
    """
    使用 pcap 文件所在的上一级目录名作为标签。

    例如：
    dataset/Benign/a.pcap -> Benign
    dataset/Attack/b.pcap -> Attack
    """
    return os.path.basename(os.path.dirname(os.path.abspath(pcap_path)))


def get_label_for_flow(flow_key, pcap_path, args, label_map=None):
    """
    根据 label_mode 获取 flow 标签。
    """
    if args.label_mode == "unknown":
        return "unknown", "unknown"

    if args.label_mode == "parent":
        return infer_label_from_path(pcap_path), "parent"

    if args.label_mode == "csv":
        if label_map is None:
            return "unknown", "csv_missing"

        label = label_map.get(flow_key, None)

        if label is None:
            return "unknown", "csv_unmatched"

        return label, "csv"

    return "unknown", "unknown"


def collect_flows_from_pcap(
    pcap_path,
    token_bytes=2,
    cut_link_layer=False,
    max_packet_tokens=None,
    max_packets_per_flow=None,
    tcp_idle_timeout=120.0,
    tcp_handshake_timeout=30.0,
    tcp_close_timeout=15.0,
    udp_idle_timeout=30.0,
):
    """
    从单个 pcap 文件中按 TCP/UDP 会话边界收集 flow。

    返回 FlowSession 列表；相同双向五元组的多次连接使用不同
    session_index，不会再合并为同一个 flow。
    """
    def tokenize(pkt):
        return packet_to_hex_tokens(
            pkt,
            token_bytes=token_bytes,
            max_packet_tokens=max_packet_tokens,
            cut_link_layer=cut_link_layer,
        )

    return collect_sessions_from_pcap(
        pcap_path=pcap_path,
        packet_to_tokens=tokenize,
        max_packets_per_flow=max_packets_per_flow,
        tcp_idle_timeout=tcp_idle_timeout,
        tcp_handshake_timeout=tcp_handshake_timeout,
        tcp_close_timeout=tcp_close_timeout,
        udp_idle_timeout=udp_idle_timeout,
    )


def packets_to_bursts(
    packet_items,
    max_burst_num=16,
    max_burst_len=128,
    min_burst_len=1,
):
    """
    将一个 flow 内的 packet 序列切成 burst。

    规则：
    同方向连续 packet 合并为一个 burst；
    方向发生变化，开启新的 burst。

    每个 burst 的 tokens 是该 burst 内多个 packet 的 token 拼接结果。
    """
    bursts = []

    current_dir = None
    current_tokens = []
    current_packet_count = 0

    for item in packet_items:
        pkt_dir = item["dir"]
        pkt_tokens = item["tokens"]

        if current_dir is None:
            current_dir = pkt_dir
            current_tokens = []
            current_packet_count = 0

        if pkt_dir != current_dir:
            if len(current_tokens) >= min_burst_len:
                bursts.append({
                    "dir": current_dir,
                    "packet_count": current_packet_count,
                    "token_count": min(len(current_tokens), max_burst_len),
                    "tokens": current_tokens[:max_burst_len],
                })

            current_dir = pkt_dir
            current_tokens = []
            current_packet_count = 0

        current_tokens.extend(pkt_tokens)
        current_packet_count += 1

    if current_dir is not None and len(current_tokens) >= min_burst_len:
        bursts.append({
            "dir": current_dir,
            "packet_count": current_packet_count,
            "token_count": min(len(current_tokens), max_burst_len),
            "tokens": current_tokens[:max_burst_len],
        })

    bursts = bursts[:max_burst_num]

    return bursts


def iter_pcap_files(input_path):
    """
    支持输入单个 pcap/pcapng 文件，也支持输入目录。

    注意：
    这里会排序，保证每次输入 pcap 顺序稳定。
    """
    if os.path.isfile(input_path):
        yield os.path.abspath(input_path)
        return

    all_files = []

    for root, dirs, files in os.walk(input_path):
        for filename in files:
            file_path = os.path.abspath(os.path.join(root, filename))
            all_files.append(file_path)
        # dirs.sort()
        # files.sort()
        #
        # for filename in files:
        #     lower_name = filename.lower()
        #
        #     if (
        #         lower_name.endswith(".pcap")
        #         or lower_name.endswith(".pcapng")
        #         or ".pcap" in lower_name
        #     ):
        #         all_files.append(os.path.abspath(os.path.join(root, filename)))

    all_files.sort()

    for p in all_files:
        yield p


def write_pcap_flows_to_part(
    flows,
    pcap_path,
    pcap_index,
    part_path,
    args,
    label_map=None,
):
    """
    将当前 pcap 已经收集好的 flows 写入一个 part JSONL 文件。

    注意：
    这个函数只处理当前一个 pcap。
    写完后，外部可以 del flows 释放内存。
    """

    written_flow_count = 0
    csv_unmatched_flow_count = 0
    local_flow_id = 0

    with open(part_path, "w", encoding="utf-8") as writer:
        for session in flows:
            flow_key = session.flow_key
            packet_items = session.packets
            bursts = packets_to_bursts(
                packet_items,
                max_burst_num=args.max_burst_num,
                max_burst_len=args.max_burst_len,
                min_burst_len=args.min_burst_len,
            )

            if len(bursts) == 0:
                continue

            if len(bursts) < args.min_burst_num:
                continue

            label, label_source = get_label_for_flow(
                flow_key=flow_key,
                pcap_path=pcap_path,
                args=args,
                label_map=label_map,
            )

            if args.label_mode == "csv" and label_source == "csv_unmatched":
                csv_unmatched_flow_count += 1

                if args.drop_unmatched_csv:
                    continue

            local_flow_id += 1
            written_flow_count += 1

            ip1, port1, ip2, port2, proto = flow_key

            sample = {
                "flow_id": f"pcap_{pcap_index:08d}_flow_{local_flow_id:08d}",
                "pcap_index": pcap_index,
                "local_flow_id": local_flow_id,
                "pcap": os.path.basename(pcap_path),
                "pcap_path": pcap_path,
                "label": label,
                "label_source": label_source,
                "session_index": session.session_index,
                "start_time": session.start_time,
                "end_time": session.end_time,
                "duration": session.duration,
                "close_reason": session.close_reason,
                "midstream": session.midstream,
                "initiator": {
                    "ip": session.initiator[0],
                    "port": session.initiator[1],
                },
                "five_tuple": {
                    "ip1": ip1,
                    "port1": port1,
                    "ip2": ip2,
                    "port2": port2,
                    "proto": proto,
                },
                "burst_num": len(bursts),
                "bursts": bursts,
            }

            writer.write(json.dumps(sample, ensure_ascii=False) + "\n")

        writer.flush()

    return {
        "written_flow_count": written_flow_count,
        "csv_unmatched_flow_count": csv_unmatched_flow_count,
    }

def process_one_pcap_to_part(task):
    """
    处理单个 pcap，并写出一个 part JSONL。

    低内存策略：
    - 只在内存中保存当前 pcap 的 flows
    - 当前 pcap 写完后立即 del flows + gc.collect()
    - 不把所有 pcap 的 flows 放到一起
    """

    global _GLOBAL_LABEL_MAP

    pcap_index = task["pcap_index"]
    pcap_path = task["pcap_path"]
    part_path = task["part_path"]
    args = SimpleNamespace(**task["args_dict"])

    result = {
        "pcap_index": pcap_index,
        "pcap_path": pcap_path,
        "part_path": part_path,
        "packet_count": 0,
        "valid_packet_count": 0,
        "skipped_packet_count": 0,
        "truncated_packet_count": 0,
        "out_of_order_packet_count": 0,
        "flow_count": 0,
        "written_flow_count": 0,
        "csv_unmatched_flow_count": 0,
        "error": None,
    }

    flows = None

    try:
        flows, stats = collect_flows_from_pcap(
            pcap_path=pcap_path,
            token_bytes=args.token_bytes,
            cut_link_layer=args.cut_link_layer,
            max_packet_tokens=args.max_packet_tokens,
            max_packets_per_flow=args.max_packets_per_flow,
            tcp_idle_timeout=getattr(args, "tcp_idle_timeout", 120.0),
            tcp_handshake_timeout=getattr(args, "tcp_handshake_timeout", 30.0),
            tcp_close_timeout=getattr(args, "tcp_close_timeout", 15.0),
            udp_idle_timeout=getattr(args, "udp_idle_timeout", 30.0),
        )

        result["packet_count"] = stats["packet_count"]
        result["valid_packet_count"] = stats["valid_packet_count"]
        result["skipped_packet_count"] = stats["skipped_packet_count"]
        result["truncated_packet_count"] = stats["truncated_packet_count"]
        result["out_of_order_packet_count"] = stats["out_of_order_packet_count"]
        result["flow_count"] = len(flows)

        write_stats = write_pcap_flows_to_part(
            flows=flows,
            pcap_path=pcap_path,
            pcap_index=pcap_index,
            part_path=part_path,
            args=args,
            label_map=_GLOBAL_LABEL_MAP,
        )

        result["written_flow_count"] = write_stats["written_flow_count"]
        result["csv_unmatched_flow_count"] = write_stats["csv_unmatched_flow_count"]

    except Exception as e:
        result["error"] = repr(e)

    finally:
        if flows is not None:
            del flows

        gc.collect()

    return result

def merge_jsonl_parts_in_order(part_infos, output_jsonl):
    """
    按 pcap_index 顺序合并 part JSONL。

    合并时重新生成全局 flow_id：
    flow_00000001
    flow_00000002
    ...
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)

    global_flow_id = 0

    part_infos = sorted(part_infos, key=lambda x: x["pcap_index"])

    with open(output_jsonl, "w", encoding="utf-8") as writer:
        for info in part_infos:
            part_path = info["part_path"]

            if not os.path.exists(part_path):
                continue

            with open(part_path, "r", encoding="utf-8") as reader:
                for line in reader:
                    line = line.strip()
                    if not line:
                        continue

                    sample = json.loads(line)

                    global_flow_id += 1
                    sample["flow_id"] = f"flow_{global_flow_id:08d}"

                    writer.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return global_flow_id

def write_flow_burst_jsonl(args):
    pcap_files = list(iter_pcap_files(args.input))

    if len(pcap_files) == 0:
        print(f"[ERROR] 没有找到 pcap 文件: {args.input}")
        return

    if getattr(args, "shuffle_pcap_files", False):
        rng = random.Random(int(getattr(args, "shuffle_seed", 42)))
        rng.shuffle(pcap_files)
        print(f"[INFO] shuffled pcap files with seed={getattr(args, 'shuffle_seed', 42)}")

    output_dir = os.path.dirname(os.path.abspath(args.output_jsonl))
    os.makedirs(output_dir, exist_ok=True)

    if args.temp_dir is None:
        temp_dir = os.path.join(output_dir, "jsonl_parts")
    else:
        temp_dir = args.temp_dir

    os.makedirs(temp_dir, exist_ok=True)

    label_map = None
    if args.label_mode == "csv":
        label_map = load_csv_labels(
            csv_path=args.label_csv,
            src_ip_col=args.csv_src_ip_col,
            src_port_col=args.csv_src_port_col,
            dst_ip_col=args.csv_dst_ip_col,
            dst_port_col=args.csv_dst_port_col,
            proto_col=args.csv_proto_col,
            label_col=args.csv_label_col,
            encoding=args.csv_encoding,
        )

    args_dict = vars(args).copy()

    tasks = []

    for idx, pcap_path in enumerate(pcap_files, start=1):
        part_path = os.path.join(temp_dir, f"part_{idx:08d}.jsonl")

        tasks.append({
            "pcap_index": idx,
            "pcap_path": pcap_path,
            "part_path": part_path,
            "args_dict": args_dict,
        })

    print(f"[INFO] pcap files: {len(tasks)}")
    print(f"[INFO] num_workers: {args.num_workers}")
    print(f"[INFO] temp_dir: {temp_dir}")

    pcap_progress_every = int(getattr(args, "pcap_progress_every", 100))
    pcap_chunksize = max(1, int(getattr(args, "pcap_chunksize", 16)))
    pcap_maxtasksperchild = getattr(args, "pcap_maxtasksperchild", None)

    print(f"[INFO] pcap_progress_every: {pcap_progress_every}")
    print(f"[INFO] pcap_chunksize: {pcap_chunksize}")
    print(f"[INFO] pcap_maxtasksperchild: {pcap_maxtasksperchild}")

    results = []

    if args.num_workers <= 1:
        init_worker(label_map)

        for task in tasks:
            result = process_one_pcap_to_part(task)
            results.append(result)

            if result["error"] is not None:
                print(f"[PCAP {result['pcap_index']}/{len(tasks)}] ERROR: {result['pcap_path']} | {result['error']}")
            elif pcap_progress_every > 0 and (
                len(results) % pcap_progress_every == 0
                or len(results) == len(tasks)
            ):
                print(
                    f"[PCAP PROGRESS {len(results)}/{len(tasks)}] "
                    f"latest={os.path.basename(result['pcap_path'])} "
                    f"packets={result['packet_count']} "
                    f"valid={result['valid_packet_count']} "
                    f"flows={result['flow_count']} "
                    f"written={result['written_flow_count']} "
                    f"csv_unmatched={result['csv_unmatched_flow_count']}"
                )
    else:
        ctx = mp.get_context("spawn")

        with ctx.Pool(
            processes=args.num_workers,
            initializer=init_worker,
            initargs=(label_map,),
            maxtasksperchild=pcap_maxtasksperchild,
        ) as pool:
            for result in pool.imap_unordered(
                process_one_pcap_to_part,
                tasks,
                chunksize=pcap_chunksize,
            ):
                results.append(result)

                idx = result["pcap_index"]

                if result["error"] is not None:
                    print(f"[PCAP {idx}/{len(tasks)}] ERROR: {result['pcap_path']} | {result['error']}")
                elif pcap_progress_every > 0 and (
                    len(results) % pcap_progress_every == 0
                    or len(results) == len(tasks)
                ):
                    print(
                        f"[PCAP PROGRESS {len(results)}/{len(tasks)}] "
                        f"latest={os.path.basename(result['pcap_path'])} "
                        f"packets={result['packet_count']}, "
                        f"valid={result['valid_packet_count']}, "
                        f"flows={result['flow_count']}, "
                        f"written={result['written_flow_count']}, "
                        f"csv_unmatched={result['csv_unmatched_flow_count']}"
                    )

    results = sorted(results, key=lambda x: x["pcap_index"])

    total_packets = sum(r["packet_count"] for r in results)
    total_valid_packets = sum(r["valid_packet_count"] for r in results)
    total_skipped_packets = sum(r["skipped_packet_count"] for r in results)
    total_truncated_packets = sum(r["truncated_packet_count"] for r in results)
    total_out_of_order_packets = sum(r["out_of_order_packet_count"] for r in results)
    total_flows = sum(r["flow_count"] for r in results)
    total_written_flows = sum(r["written_flow_count"] for r in results)
    total_csv_unmatched = sum(r["csv_unmatched_flow_count"] for r in results)
    total_errors = sum(1 for r in results if r["error"] is not None)

    print("\n[SUMMARY]")
    print(f"pcap files: {len(results)}")
    print(f"errors: {total_errors}")
    print(f"total packets: {total_packets}")
    print(f"valid tcp/udp packets: {total_valid_packets}")
    print(f"skipped packets: {total_skipped_packets}")
    print(f"truncated packets: {total_truncated_packets}")
    print(f"out-of-order timestamps: {total_out_of_order_packets}")
    print(f"total flows: {total_flows}")
    print(f"written flows: {total_written_flows}")

    if args.label_mode == "csv":
        print(f"csv unmatched flows: {total_csv_unmatched}")

    print(f"parts dir: {temp_dir}")

    if args.merge_parts:
        print("\n[MERGE] merging parts in pcap order...")

        merged_count = merge_jsonl_parts_in_order(
            part_infos=results,
            output_jsonl=args.output_jsonl,
        )

        print(f"[MERGE] merged flows: {merged_count}")
        print(f"[MERGE] output: {args.output_jsonl}")

        if not args.keep_parts:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"[MERGE] removed temp_dir: {temp_dir}")
    else:
        print("\n[INFO] 未合并分片。")
        print("你可以直接使用每个 part_xxxxxxxx.jsonl，或者之后再按编号合并。")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PCAP/PCAPNG files to flow-burst JSONL format."
    )

    parser.add_argument(
        "--input",
        type=str,
        # required=True,
        default="C:/ET-Test/ET-B-TEST/datasets/pcap1",
        help="输入 pcap 文件或 pcap 目录",
    )

    parser.add_argument(
        "--output_jsonl",
        type=str,
        # required=True,
        default="C:/HierarchicalBurst-FlowBERT/datasets/pre_train_output_jsonl/flow_burst_labeled.jsonl",
        help="输出 flow_burst.jsonl 路径",
    )

    parser.add_argument(
        "--token_bytes",
        type=int,
        default=2,
        help="每个 token 包含几个字节，默认 2，即 4 个 hex 字符",
    )

    parser.add_argument(
        "--max_burst_num",
        type=int,
        default=16,
        help="每个 flow 最多保留多少个 burst",
    )

    parser.add_argument(
        "--max_burst_len",
        type=int,
        default=128,
        help="每个 burst 最多保留多少个 token",
    )

    parser.add_argument(
        "--min_burst_len",
        type=int,
        default=1,
        help="过滤 token 数小于该值的 burst",
    )

    parser.add_argument(
        "--min_burst_num",
        type=int,
        default=1,
        help="过滤 burst 数小于该值的 flow",
    )

    parser.add_argument(
        "--max_packet_tokens",
        type=int,
        default=None,
        help="每个 packet 最多保留多少 token，默认不限制",
    )

    parser.add_argument(
        "--max_packets_per_flow",
        type=int,
        default=None,
        help="每个 flow 最多保留多少 packet，默认不限制",
    )

    parser.add_argument("--tcp_idle_timeout", type=float, default=120.0)
    parser.add_argument("--tcp_handshake_timeout", type=float, default=30.0)
    parser.add_argument("--tcp_close_timeout", type=float, default=15.0)
    parser.add_argument("--udp_idle_timeout", type=float, default=30.0)

    parser.add_argument(
        "--cut_link_layer",
        action="store_true",
        default=True,
        help="是否从 IP 层开始取 bytes，开启后会去掉以太网链路层头",
    )

    parser.add_argument(
        "--label_mode",
        type=str,
        default="unknown",
        choices=["unknown", "parent", "csv"],
        help="标签模式：unknown 无标签；parent 使用上级目录名；csv 根据 CSV 五元组匹配",
    )

    parser.add_argument(
        "--label_csv",
        type=str,
        default=None,
        help="标签 CSV 文件路径，仅 label_mode=csv 时需要",
    )

    parser.add_argument(
        "--csv_encoding",
        type=str,
        default="utf-8-sig",
        help="CSV 文件编码，默认 utf-8-sig",
    )

    parser.add_argument(
        "--csv_src_ip_col",
        type=str,
        default="Source IP",
        help="CSV 中源 IP 列名",
    )

    parser.add_argument(
        "--csv_src_port_col",
        type=str,
        default="Source Port",
        help="CSV 中源端口列名",
    )

    parser.add_argument(
        "--csv_dst_ip_col",
        type=str,
        default="Destination IP",
        help="CSV 中目的 IP 列名",
    )

    parser.add_argument(
        "--csv_dst_port_col",
        type=str,
        default="Destination Port",
        help="CSV 中目的端口列名",
    )

    parser.add_argument(
        "--csv_proto_col",
        type=str,
        default="Protocol",
        help="CSV 中协议列名",
    )

    parser.add_argument(
        "--csv_label_col",
        type=str,
        default="Label",
        help="CSV 中标签列名",
    )

    parser.add_argument(
        "--drop_unmatched_csv",
        action="store_true",
        help="label_mode=csv 时，是否丢弃 CSV 中匹配不到标签的 flow",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="并行处理 pcap 的进程数，默认 1。建议设置为 CPU 核数的一半或磁盘能承受的数量",
    )

    parser.add_argument(
        "--pcap_progress_every",
        type=int,
        default=100,
        help="每处理多少个 pcap 打印一次进度；0 表示只打印错误和汇总",
    )

    parser.add_argument(
        "--pcap_chunksize",
        type=int,
        default=16,
        help="多进程任务批量分发大小。小 pcap 很多时建议 16/32/64",
    )

    parser.add_argument(
        "--pcap_maxtasksperchild",
        type=int,
        default=None,
        help="每个 worker 处理多少个 pcap 后重启。默认不重启；小 pcap 很多时不要设为 1",
    )

    parser.add_argument(
        "--shuffle_pcap_files",
        action="store_true",
        help="在生成 JSONL part 前打乱 pcap 文件顺序，让样本分布更随机",
    )

    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="shuffle_pcap_files 使用的随机种子",
    )

    parser.add_argument(
        "--temp_dir",
        type=str,
        default=None,
        help="临时 JSONL 分片目录，默认使用 output_jsonl 同目录下的 jsonl_parts",
    )

    parser.add_argument(
        "--merge_parts",
        action="store_true",
        default=True,
        help="是否将多个 part JSONL 按 pcap 原始顺序合并成 output_jsonl",
    )

    parser.add_argument(
        "--keep_parts",
        action="store_true",
        default=True,
        help="合并完成后是否保留分片 JSONL 文件",
    )

    args = parser.parse_args()
    write_flow_burst_jsonl(args)


if __name__ == "__main__":
    main()
