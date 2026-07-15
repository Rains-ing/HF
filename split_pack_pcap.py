#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_pack_pcap_dir_by_flow_size_keep_big.py

功能：
1. 支持输入单个 pcap/pcapng 文件，也支持输入一个目录。
2. 按五元组把 packet 归并成 flow。
3. 先把每个完整 flow 写到临时 pcap。
4. 再把多个完整 flow 打包成 15~35MB 左右的小 pcap。
5. 如果某个单独 flow 本身超过 max_mb，例如 35MB：
   不切断 flow，单独输出成一个超过 35MB 的 pcap，并标记为 OVERSIZED_SINGLE_FLOW。

目录输入时有两种模式：
- --directory_mode combine：默认，把目录里所有 pcap 的 flow 混合后统一打包。
- --directory_mode per_file：每个原始 pcap 单独处理，输出到对应子目录。

默认使用双向 flow/session：
A:port -> B:port 和 B:port -> A:port 会放到同一个 flow。
如果你想严格单向五元组，使用 --mode uni。

依赖：
    pip install scapy

目录输入推荐运行：
    python split_pack_pcap_dir_by_flow_size_keep_big.py \
      -i /path/to/pcap_dir \
      -o /path/to/output_chunks \
      --tmp_dir /path/to/tmp_flow_pcaps \
      --directory_mode combine \
      --mode bidi \
      --min_mb 15 \
      --max_mb 35 \
      --target_mb 25
"""

import os
import re
import csv
import time
import shutil
import hashlib
import argparse
import random
from collections import OrderedDict

from scapy.all import PcapReader, PcapWriter, IP, IPv6, TCP, UDP


PCAP_EXTS = {".pcap", ".pcapng", ".cap"}


def mb_to_bytes(x: float) -> int:
    return int(float(x) * 1024 * 1024)


def bytes_to_mb(x: int) -> float:
    return float(x) / 1024 / 1024


def safe_name(text: str, max_len: int = 120) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = text.strip("._-") or "unnamed"
    if len(text) > max_len:
        text = text[:max_len]
    return text


def is_pcap_file(path: str) -> bool:
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in PCAP_EXTS


def find_pcap_files(input_path: str, recursive: bool = True):
    """输入可以是单文件或目录，返回 pcap/pcapng/cap 文件列表。"""
    input_path = os.path.abspath(input_path)

    if os.path.isfile(input_path):
        if not is_pcap_file(input_path):
            raise ValueError(f"输入文件扩展名不是 pcap/pcapng/cap: {input_path}")
        return [input_path]

    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    pcaps = []
    if recursive:
        for root, _, files in os.walk(input_path):
            for name in files:
                path = os.path.join(root, name)
                if is_pcap_file(path):
                    pcaps.append(os.path.abspath(path))
    else:
        for name in os.listdir(input_path):
            path = os.path.join(input_path, name)
            if is_pcap_file(path):
                pcaps.append(os.path.abspath(path))

    pcaps.sort()
    return pcaps


def source_tag_from_path(path: str, input_root: str = None, index: int = None) -> str:
    """为每个源 pcap 生成唯一 tag，避免目录模式下不同文件的相同五元组混在一起。"""
    abs_path = os.path.abspath(path)

    if input_root and os.path.isdir(input_root):
        rel = os.path.relpath(abs_path, os.path.abspath(input_root))
    else:
        rel = os.path.basename(abs_path)

    digest = hashlib.md5(abs_path.encode("utf-8")).hexdigest()[:10]
    prefix = f"{index:06d}_" if index is not None else ""
    stem = os.path.splitext(rel)[0].replace(os.sep, "__")
    return safe_name(f"{prefix}{stem}_{digest}", max_len=100)


def get_flow_key(pkt, mode="bidi", tcp_udp_only=True):
    """
    从 packet 中提取 flow key。

    mode:
        bidi: 双向 session，A->B 和 B->A 归为同一个 flow。
        uni : 严格单向五元组，A->B 和 B->A 分开。

    返回：
        (ip_version, protocol, src_ip, src_port, dst_ip, dst_port)
    """
    if IP in pkt:
        ip = pkt[IP]
        src_ip = ip.src
        dst_ip = ip.dst
        ip_version = "IPv4"
        proto_num = int(ip.proto)
    elif IPv6 in pkt:
        ip = pkt[IPv6]
        src_ip = ip.src
        dst_ip = ip.dst
        ip_version = "IPv6"
        proto_num = int(ip.nh)
    else:
        return None

    if TCP in pkt:
        proto = "TCP"
        src_port = int(pkt[TCP].sport)
        dst_port = int(pkt[TCP].dport)
    elif UDP in pkt:
        proto = "UDP"
        src_port = int(pkt[UDP].sport)
        dst_port = int(pkt[UDP].dport)
    else:
        if tcp_udp_only:
            return None
        proto = f"PROTO{proto_num}"
        src_port = 0
        dst_port = 0

    if mode == "uni":
        return (ip_version, proto, src_ip, src_port, dst_ip, dst_port)

    endpoint_a = (src_ip, src_port)
    endpoint_b = (dst_ip, dst_port)

    if endpoint_a <= endpoint_b:
        return (ip_version, proto, endpoint_a[0], endpoint_a[1], endpoint_b[0], endpoint_b[1])
    return (ip_version, proto, endpoint_b[0], endpoint_b[1], endpoint_a[0], endpoint_a[1])


def flow_key_to_temp_path(flow_key, tmp_dir, source_tag=""):
    ip_version, proto, src_ip, src_port, dst_ip, dst_port = flow_key
    raw = f"{ip_version}_{proto}_{src_ip}_{src_port}__{dst_ip}_{dst_port}"
    hash_raw = f"{source_tag}__{raw}" if source_tag else raw
    digest = hashlib.md5(hash_raw.encode("utf-8")).hexdigest()

    if source_tag:
        name = f"{safe_name(source_tag, 50)}_{digest[:16]}_{safe_name(raw, 80)}.pcap"
    else:
        name = f"{digest[:16]}_{safe_name(raw, 100)}.pcap"

    subdir = digest[:2]
    return os.path.join(tmp_dir, subdir, name)


class LRUWriterCache:
    """用 LRU 缓存控制同时打开的 PcapWriter 数量，防止文件描述符过多。"""

    def __init__(self, max_open_files=256):
        self.max_open_files = int(max_open_files)
        self.cache = OrderedDict()

    def write(self, path, pkt):
        writer = self.cache.pop(path, None)

        if writer is None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            append = os.path.exists(path)
            writer = PcapWriter(path, append=append, sync=False)

        writer.write(pkt)
        self.cache[path] = writer

        if len(self.cache) > self.max_open_files:
            _, old_writer = self.cache.popitem(last=False)
            old_writer.close()

    def close_all(self):
        for writer in self.cache.values():
            writer.close()
        self.cache.clear()


def split_to_temp_flow_pcaps(
    input_pcap,
    tmp_dir,
    mode="bidi",
    tcp_udp_only=True,
    max_open_files=256,
    log_interval=100000,
    source_tag="",
):
    """第一步：把一个源 pcap 按 flow 写成临时 pcap。"""
    os.makedirs(tmp_dir, exist_ok=True)
    writer_cache = LRUWriterCache(max_open_files=max_open_files)

    flow_info = {}
    total_packets = 0
    valid_packets = 0
    written_packets = 0
    skipped_packets = 0
    t0 = time.time()

    try:
        with PcapReader(input_pcap) as reader:
            for pkt in reader:
                total_packets += 1

                flow_key = get_flow_key(pkt, mode=mode, tcp_udp_only=tcp_udp_only)
                if flow_key is None:
                    skipped_packets += 1
                    continue

                valid_packets += 1
                temp_path = flow_key_to_temp_path(flow_key, tmp_dir, source_tag=source_tag)
                writer_cache.write(temp_path, pkt)

                unique_key = (source_tag, flow_key)
                if unique_key not in flow_info:
                    flow_info[unique_key] = {
                        "flow_key": flow_key,
                        "source_tag": source_tag,
                        "source_pcap": input_pcap,
                        "path": temp_path,
                        "packet_count": 0,
                        "first_seen_index": total_packets,
                    }

                flow_info[unique_key]["packet_count"] += 1
                written_packets += 1

                if log_interval > 0 and total_packets % log_interval == 0:
                    elapsed = time.time() - t0
                    pps = total_packets / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[SPLIT] source={os.path.basename(input_pcap)} "
                        f"packets={total_packets} valid={valid_packets} written={written_packets} "
                        f"flows={len(flow_info)} skipped={skipped_packets} "
                        f"speed={pps:.1f} pkt/s elapsed={elapsed/3600:.2f}h",
                        flush=True,
                    )
    finally:
        writer_cache.close_all()

    elapsed = time.time() - t0
    print("\n[SPLIT DONE]")
    print(f"input_pcap    : {input_pcap}")
    print(f"source_tag    : {source_tag}")
    print(f"tmp_dir       : {tmp_dir}")
    print(f"total_packets : {total_packets}")
    print(f"valid_packets : {valid_packets}")
    print(f"written       : {written_packets}")
    print(f"flows         : {len(flow_info)}")
    print(f"skipped       : {skipped_packets}")
    print(f"elapsed       : {elapsed/3600:.2f}h")
    print()

    return flow_info


def build_flow_items(flow_info):
    items = []
    missing = 0

    for _, info in flow_info.items():
        path = info["path"]
        if not os.path.exists(path):
            missing += 1
            continue

        size = os.path.getsize(path)
        if size <= 0:
            continue

        items.append(
            {
                "flow_key": info["flow_key"],
                "source_tag": info.get("source_tag", ""),
                "source_pcap": info.get("source_pcap", ""),
                "path": path,
                "packet_count": info["packet_count"],
                "first_seen_index": info["first_seen_index"],
                "size": size,
            }
        )

    if missing:
        print(f"[WARN] missing temp flow files: {missing}")

    return items


def sort_flow_items(items, pack_order, seed=42):
    items = list(items)

    if pack_order == "size_desc":
        items.sort(key=lambda x: x["size"], reverse=True)
    elif pack_order == "size_asc":
        items.sort(key=lambda x: x["size"])
    elif pack_order == "original":
        items.sort(key=lambda x: (x.get("source_tag", ""), x["first_seen_index"]))
    elif pack_order == "random":
        rnd = random.Random(seed)
        rnd.shuffle(items)
    else:
        raise ValueError(f"Unknown pack_order: {pack_order}")

    return items


def pack_flows_to_chunks(flow_items, min_bytes, max_bytes, target_bytes, pack_order="size_desc", seed=42):
    """
    第二步：把 flow 打包成 chunk。
    - flow.size > max_bytes 的 flow 不切断，单独作为一个 chunk 输出。
    - 普通 flow 尽量打包到 target_bytes 附近，并且不超过 max_bytes。
    """
    oversized_items = []
    normal_items = []

    for item in flow_items:
        if item["size"] > max_bytes:
            oversized_items.append(item)
        else:
            normal_items.append(item)

    normal_items = sort_flow_items(normal_items, pack_order=pack_order, seed=seed)

    chunks = []
    current = []
    current_size = 0

    for item in normal_items:
        item_size = item["size"]

        if current and current_size + item_size > max_bytes:
            chunks.append(
                {
                    "items": current,
                    "estimated_size": current_size,
                    "status": "OK" if current_size >= min_bytes else "UNDER_MIN",
                }
            )
            current = []
            current_size = 0

        current.append(item)
        current_size += item_size

        if current_size >= target_bytes:
            chunks.append(
                {
                    "items": current,
                    "estimated_size": current_size,
                    "status": "OK" if current_size >= min_bytes else "UNDER_MIN",
                }
            )
            current = []
            current_size = 0

    if current:
        if chunks and chunks[-1]["estimated_size"] + current_size <= max_bytes:
            chunks[-1]["items"].extend(current)
            chunks[-1]["estimated_size"] += current_size
            chunks[-1]["status"] = "OK" if chunks[-1]["estimated_size"] >= min_bytes else "UNDER_MIN"
        else:
            chunks.append(
                {
                    "items": current,
                    "estimated_size": current_size,
                    "status": "OK" if current_size >= min_bytes else "UNDER_MIN",
                }
            )

    oversized_items = sort_flow_items(oversized_items, pack_order="size_desc", seed=seed)
    for item in oversized_items:
        chunks.append(
            {
                "items": [item],
                "estimated_size": item["size"],
                "status": "OVERSIZED_SINGLE_FLOW",
            }
        )

    return chunks


def read_packets_from_flow_pcaps(chunk_items):
    packets = []
    for item in chunk_items:
        with PcapReader(item["path"]) as reader:
            for pkt in reader:
                packets.append(pkt)
    return packets


def write_chunk_pcap(chunk_items, output_pcap, sort_packets=True):
    """
    写一个输出 chunk pcap。
    对普通 chunk，默认按 packet 时间排序。
    对单个 flow，直接顺序写，避免大 flow 一次性读入内存。
    """
    os.makedirs(os.path.dirname(output_pcap), exist_ok=True)

    packet_count = 0
    writer = None

    try:
        if sort_packets and len(chunk_items) > 1:
            packets = read_packets_from_flow_pcaps(chunk_items)
            packets.sort(key=lambda p: float(getattr(p, "time", 0.0)))

            writer = PcapWriter(output_pcap, sync=False)
            for pkt in packets:
                writer.write(pkt)
                packet_count += 1
        else:
            writer = PcapWriter(output_pcap, sync=False)
            for item in chunk_items:
                with PcapReader(item["path"]) as reader:
                    for pkt in reader:
                        writer.write(pkt)
                        packet_count += 1
    finally:
        if writer is not None:
            writer.close()

    actual_size = os.path.getsize(output_pcap) if os.path.exists(output_pcap) else 0
    return actual_size, packet_count


def write_manifests(output_dir, chunks, chunk_records):
    chunk_manifest = os.path.join(output_dir, "chunk_manifest.csv")
    flow_manifest = os.path.join(output_dir, "chunk_flow_manifest.csv")

    with open(chunk_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "chunk_id",
                "pcap_path",
                "flow_count",
                "packet_count",
                "estimated_size_mb",
                "actual_size_mb",
                "status",
            ]
        )

        for rec in chunk_records:
            writer.writerow(
                [
                    rec["chunk_id"],
                    rec["pcap_path"],
                    rec["flow_count"],
                    rec["packet_count"],
                    f"{bytes_to_mb(rec['estimated_size']):.3f}",
                    f"{bytes_to_mb(rec['actual_size']):.3f}",
                    rec["status"],
                ]
            )

    with open(flow_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "chunk_id",
                "chunk_status",
                "source_tag",
                "source_pcap",
                "ip_version",
                "protocol",
                "src_ip",
                "src_port",
                "dst_ip",
                "dst_port",
                "flow_packet_count",
                "flow_size_mb",
                "temp_flow_pcap",
            ]
        )

        for chunk_id, chunk in enumerate(chunks, start=1):
            for item in chunk["items"]:
                ip_version, proto, src_ip, src_port, dst_ip, dst_port = item["flow_key"]
                writer.writerow(
                    [
                        chunk_id,
                        chunk["status"],
                        item.get("source_tag", ""),
                        item.get("source_pcap", ""),
                        ip_version,
                        proto,
                        src_ip,
                        src_port,
                        dst_ip,
                        dst_port,
                        item["packet_count"],
                        f"{bytes_to_mb(item['size']):.3f}",
                        item["path"],
                    ]
                )

    print(f"chunk_manifest      : {chunk_manifest}")
    print(f"chunk_flow_manifest : {flow_manifest}")


def pack_and_write_flow_items(
    flow_items,
    output_dir,
    min_mb=15,
    max_mb=35,
    target_mb=25,
    pack_order="size_desc",
    seed=42,
    drop_under_min=False,
    sort_packets=True,
):
    os.makedirs(output_dir, exist_ok=True)

    min_bytes = mb_to_bytes(min_mb)
    max_bytes = mb_to_bytes(max_mb)
    target_bytes = mb_to_bytes(target_mb)

    if not (0 < min_bytes <= target_bytes <= max_bytes):
        raise ValueError("必须满足：0 < min_mb <= target_mb <= max_mb")

    total_flow_bytes = sum(x["size"] for x in flow_items)
    oversized_count = sum(1 for x in flow_items if x["size"] > max_bytes)

    print("[FLOW SUMMARY]")
    print(f"flow_count          : {len(flow_items)}")
    print(f"total_temp_size     : {bytes_to_mb(total_flow_bytes):.2f}MB")
    print(f"oversized_flow_count: {oversized_count}")
    print()

    chunks = pack_flows_to_chunks(
        flow_items=flow_items,
        min_bytes=min_bytes,
        max_bytes=max_bytes,
        target_bytes=target_bytes,
        pack_order=pack_order,
        seed=seed,
    )

    if drop_under_min:
        before = len(chunks)
        chunks = [c for c in chunks if c["status"] != "UNDER_MIN"]
        print(f"[DROP] UNDER_MIN chunks dropped: {before - len(chunks)}")

    print("[PACK SUMMARY]")
    print(f"chunk_count: {len(chunks)}")
    print()

    chunk_records = []
    t0 = time.time()

    for idx, chunk in enumerate(chunks, start=1):
        if chunk["status"] == "OVERSIZED_SINGLE_FLOW":
            output_pcap = os.path.join(output_dir, f"chunk_{idx:06d}_OVERSIZED_SINGLE_FLOW.pcap")
        elif chunk["status"] == "UNDER_MIN":
            output_pcap = os.path.join(output_dir, f"chunk_{idx:06d}_UNDER_MIN.pcap")
        else:
            output_pcap = os.path.join(output_dir, f"chunk_{idx:06d}.pcap")

        actual_size, packet_count = write_chunk_pcap(
            chunk_items=chunk["items"],
            output_pcap=output_pcap,
            sort_packets=sort_packets,
        )

        rec = {
            "chunk_id": idx,
            "pcap_path": output_pcap,
            "flow_count": len(chunk["items"]),
            "packet_count": packet_count,
            "estimated_size": chunk["estimated_size"],
            "actual_size": actual_size,
            "status": chunk["status"],
        }
        chunk_records.append(rec)

        print(
            f"[WRITE {idx}/{len(chunks)}] "
            f"flows={rec['flow_count']} packets={packet_count} "
            f"actual={bytes_to_mb(actual_size):.2f}MB "
            f"status={chunk['status']} path={output_pcap}",
            flush=True,
        )

    elapsed = time.time() - t0

    print("\n[DONE]")
    print(f"output_dir: {output_dir}")
    print(f"write_elapsed: {elapsed/3600:.2f}h")

    ok_count = sum(1 for r in chunk_records if r["status"] == "OK")
    under_count = sum(1 for r in chunk_records if r["status"] == "UNDER_MIN")
    over_count = sum(1 for r in chunk_records if r["status"] == "OVERSIZED_SINGLE_FLOW")

    print(f"OK chunks              : {ok_count}")
    print(f"UNDER_MIN chunks       : {under_count}")
    print(f"OVERSIZED_SINGLE_FLOW  : {over_count}")
    print()

    write_manifests(output_dir, chunks, chunk_records)

    return chunk_records


def split_and_pack_one_file(
    input_pcap,
    output_dir,
    tmp_dir,
    mode="bidi",
    min_mb=15,
    max_mb=35,
    target_mb=25,
    tcp_udp_only=True,
    max_open_files=256,
    log_interval=100000,
    pack_order="size_desc",
    seed=42,
    drop_under_min=False,
    sort_packets=True,
    clean_tmp=False,
    source_tag="",
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    print("[CONFIG]")
    print(f"input_pcap       : {input_pcap}")
    print(f"output_dir       : {output_dir}")
    print(f"tmp_dir          : {tmp_dir}")
    print(f"mode             : {mode}")
    print(f"size range       : {min_mb}MB ~ {max_mb}MB")
    print(f"target size      : {target_mb}MB")
    print(f"tcp_udp_only     : {tcp_udp_only}")
    print(f"pack_order       : {pack_order}")
    print(f"oversized policy : keep as single pcap, may be > {max_mb}MB")
    print(f"sort_packets     : {sort_packets}")
    print()

    flow_info = split_to_temp_flow_pcaps(
        input_pcap=input_pcap,
        tmp_dir=tmp_dir,
        mode=mode,
        tcp_udp_only=tcp_udp_only,
        max_open_files=max_open_files,
        log_interval=log_interval,
        source_tag=source_tag,
    )

    flow_items = build_flow_items(flow_info)

    records = pack_and_write_flow_items(
        flow_items=flow_items,
        output_dir=output_dir,
        min_mb=min_mb,
        max_mb=max_mb,
        target_mb=target_mb,
        pack_order=pack_order,
        seed=seed,
        drop_under_min=drop_under_min,
        sort_packets=sort_packets,
    )

    if clean_tmp:
        print(f"[CLEAN] removing tmp_dir: {tmp_dir}")
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return records


def split_and_pack_combine_inputs(
    input_pcaps,
    output_dir,
    tmp_dir,
    input_root=None,
    mode="bidi",
    min_mb=15,
    max_mb=35,
    target_mb=25,
    tcp_udp_only=True,
    max_open_files=256,
    log_interval=100000,
    pack_order="size_desc",
    seed=42,
    drop_under_min=False,
    sort_packets=True,
    clean_tmp=False,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    print("[DIRECTORY CONFIG]")
    print(f"pcap_count       : {len(input_pcaps)}")
    print(f"output_dir       : {output_dir}")
    print(f"tmp_dir          : {tmp_dir}")
    print(f"directory_mode   : combine")
    print(f"mode             : {mode}")
    print(f"size range       : {min_mb}MB ~ {max_mb}MB")
    print(f"target size      : {target_mb}MB")
    print(f"oversized policy : keep as single pcap, may be > {max_mb}MB")
    print()

    all_flow_info = {}

    for idx, pcap in enumerate(input_pcaps, start=1):
        tag = source_tag_from_path(pcap, input_root=input_root, index=idx)
        print(f"\n========== [SOURCE {idx}/{len(input_pcaps)}] {pcap} ==========")
        flow_info = split_to_temp_flow_pcaps(
            input_pcap=pcap,
            tmp_dir=tmp_dir,
            mode=mode,
            tcp_udp_only=tcp_udp_only,
            max_open_files=max_open_files,
            log_interval=log_interval,
            source_tag=tag,
        )
        all_flow_info.update(flow_info)

    flow_items = build_flow_items(all_flow_info)

    records = pack_and_write_flow_items(
        flow_items=flow_items,
        output_dir=output_dir,
        min_mb=min_mb,
        max_mb=max_mb,
        target_mb=target_mb,
        pack_order=pack_order,
        seed=seed,
        drop_under_min=drop_under_min,
        sort_packets=sort_packets,
    )

    if clean_tmp:
        print(f"[CLEAN] removing tmp_dir: {tmp_dir}")
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return records


def write_directory_summary(output_dir, rows):
    path = os.path.join(output_dir, "directory_summary.csv")
    os.makedirs(output_dir, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_pcap", "source_tag", "output_dir", "tmp_dir", "status"])
        writer.writerows(rows)

    print(f"directory_summary : {path}")


def main():
    parser = argparse.ArgumentParser(
        description="输入单个 pcap 或 pcap 目录，按五元组拆分 flow，并把完整 flow 打包成 15~35MB 的小 pcap；超过 35MB 的单 flow 单独保留。"
    )

    parser.add_argument(
        "-i",
        "--input_pcap",
        required=True,
        help="输入路径：可以是单个 pcap/pcapng/cap 文件，也可以是包含多个 pcap 的目录",
    )
    parser.add_argument("-o", "--output_dir", required=True, help="输出 chunk pcap 的目录")
    parser.add_argument("--tmp_dir", required=True, help="临时 flow pcap 目录，建议放到高速磁盘")

    parser.add_argument(
        "--directory_mode",
        choices=["combine", "per_file"],
        default="combine",
        help="当 -i 是目录时：combine=所有pcap的flow混合后统一打包；per_file=每个源pcap单独处理。默认 combine。",
    )
    parser.add_argument(
        "--no_recursive",
        action="store_true",
        help="当 -i 是目录时，默认递归查找 pcap；加此参数后只处理当前目录第一层。",
    )

    parser.add_argument(
        "--mode",
        choices=["bidi", "uni"],
        default="bidi",
        help="bidi=双向flow/session；uni=严格单向五元组flow。默认 bidi。",
    )

    parser.add_argument("--min_mb", type=float, default=15, help="输出 pcap 的最小目标大小，默认 15MB")
    parser.add_argument("--max_mb", type=float, default=35, help="输出 pcap 的最大目标大小，默认 35MB")
    parser.add_argument("--target_mb", type=float, default=25, help="打包目标大小，默认 25MB")

    parser.add_argument(
        "--include_non_tcp_udp",
        action="store_true",
        help="默认只处理 TCP/UDP；加上这个参数后也保留非 TCP/UDP 的 IP 包，端口记为 0。",
    )

    parser.add_argument("--max_open_files", type=int, default=256, help="最多同时打开多少个临时 flow pcap 文件")
    parser.add_argument("--log_interval", type=int, default=100000, help="每处理多少个 packet 打印一次进度")

    parser.add_argument(
        "--pack_order",
        choices=["size_desc", "size_asc", "original", "random"],
        default="size_desc",
        help="flow 打包顺序。默认 size_desc。",
    )

    parser.add_argument("--seed", type=int, default=42, help="pack_order=random 时使用的随机种子")
    parser.add_argument("--drop_under_min", action="store_true", help="丢弃最后不足 min_mb 的 UNDER_MIN chunk")
    parser.add_argument("--no_sort_packets", action="store_true", help="默认会对多 flow chunk 按时间排序；加此参数则不排序，直接按 flow 拼接")
    parser.add_argument("--clean_tmp", action="store_true", help="处理完成后删除临时 flow pcap 目录")

    args = parser.parse_args()

    input_path = os.path.abspath(args.input_pcap)
    output_dir = os.path.abspath(args.output_dir)
    tmp_dir = os.path.abspath(args.tmp_dir)
    recursive = not args.no_recursive

    input_pcaps = find_pcap_files(input_path, recursive=recursive)

    if len(input_pcaps) == 0:
        raise RuntimeError(f"没有在输入路径中找到 pcap/pcapng/cap 文件: {input_path}")

    print("[INPUT]")
    print(f"input_path : {input_path}")
    print(f"pcap_count : {len(input_pcaps)}")
    for idx, p in enumerate(input_pcaps[:10], start=1):
        print(f"  [{idx}] {p}")
    if len(input_pcaps) > 10:
        print(f"  ... and {len(input_pcaps) - 10} more")
    print()

    if len(input_pcaps) == 1:
        tag = source_tag_from_path(input_pcaps[0], input_root=input_path, index=1)
        split_and_pack_one_file(
            input_pcap=input_pcaps[0],
            output_dir=output_dir,
            tmp_dir=tmp_dir,
            mode=args.mode,
            min_mb=args.min_mb,
            max_mb=args.max_mb,
            target_mb=args.target_mb,
            tcp_udp_only=not args.include_non_tcp_udp,
            max_open_files=args.max_open_files,
            log_interval=args.log_interval,
            pack_order=args.pack_order,
            seed=args.seed,
            drop_under_min=args.drop_under_min,
            sort_packets=not args.no_sort_packets,
            clean_tmp=args.clean_tmp,
            source_tag=tag,
        )
        return

    if args.directory_mode == "combine":
        split_and_pack_combine_inputs(
            input_pcaps=input_pcaps,
            output_dir=output_dir,
            tmp_dir=tmp_dir,
            input_root=input_path,
            mode=args.mode,
            min_mb=args.min_mb,
            max_mb=args.max_mb,
            target_mb=args.target_mb,
            tcp_udp_only=not args.include_non_tcp_udp,
            max_open_files=args.max_open_files,
            log_interval=args.log_interval,
            pack_order=args.pack_order,
            seed=args.seed,
            drop_under_min=args.drop_under_min,
            sort_packets=not args.no_sort_packets,
            clean_tmp=args.clean_tmp,
        )
    else:
        rows = []
        for idx, pcap in enumerate(input_pcaps, start=1):
            tag = source_tag_from_path(pcap, input_root=input_path, index=idx)
            out_subdir = os.path.join(output_dir, tag)
            tmp_subdir = os.path.join(tmp_dir, tag)

            print(f"\n========== [PER_FILE {idx}/{len(input_pcaps)}] {pcap} ==========")
            try:
                split_and_pack_one_file(
                    input_pcap=pcap,
                    output_dir=out_subdir,
                    tmp_dir=tmp_subdir,
                    mode=args.mode,
                    min_mb=args.min_mb,
                    max_mb=args.max_mb,
                    target_mb=args.target_mb,
                    tcp_udp_only=not args.include_non_tcp_udp,
                    max_open_files=args.max_open_files,
                    log_interval=args.log_interval,
                    pack_order=args.pack_order,
                    seed=args.seed,
                    drop_under_min=args.drop_under_min,
                    sort_packets=not args.no_sort_packets,
                    clean_tmp=args.clean_tmp,
                    source_tag=tag,
                )
                rows.append([pcap, tag, out_subdir, tmp_subdir, "OK"])
            except Exception as e:
                rows.append([pcap, tag, out_subdir, tmp_subdir, f"ERROR: {repr(e)}"])
                print(f"[ERROR] failed on {pcap}: {repr(e)}", flush=True)

        write_directory_summary(output_dir, rows)


if __name__ == "__main__":
    main()
