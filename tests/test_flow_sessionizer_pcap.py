import os
import tempfile
from types import SimpleNamespace
import unittest
import json


try:
    from scapy.all import Ether, IP, TCP, UDP, Raw, wrpcap
    from process_pretrain_data.pcap_to_flow_burst_json import (
        collect_flows_from_pcap as collect_pretrain_flows,
        write_pcap_flows_to_part,
    )
    from process_fine_tuning.pcap_to_flow_burst_jsonl_zero_ip_port import (
        collect_flows_from_pcap as collect_finetune_flows,
    )
    SCAPY_AVAILABLE = True
except ModuleNotFoundError:
    SCAPY_AVAILABLE = False


def stamp(packet, timestamp):
    packet.time = timestamp
    return packet


@unittest.skipUnless(SCAPY_AVAILABLE, "Scapy is required for PCAP integration tests")
class FlowSessionizerPcapTest(unittest.TestCase):
    def build_pcap(self, path):
        client = "10.0.0.1"
        server = "10.0.0.2"
        sport = 50000
        dport = 443

        packets = [
            stamp(Ether() / IP(src=client, dst=server) / TCP(sport=sport, dport=dport, flags="S", seq=100), 1.0),
            stamp(Ether() / IP(src=client, dst=server) / TCP(sport=sport, dport=dport, flags="S", seq=100), 1.1),
            stamp(Ether() / IP(src=server, dst=client) / TCP(sport=dport, dport=sport, flags="SA", seq=500, ack=101), 1.2),
            stamp(Ether() / IP(src=client, dst=server) / TCP(sport=sport, dport=dport, flags="A", seq=101, ack=501), 1.3),
            stamp(Ether() / IP(src=client, dst=server) / TCP(sport=sport, dport=dport, flags="PA", seq=101, ack=501) / Raw(b"first"), 1.4),
            stamp(Ether() / IP(src=server, dst=client) / TCP(sport=dport, dport=sport, flags="RA", seq=501, ack=106), 1.5),
            stamp(Ether() / IP(src=client, dst=server) / TCP(sport=sport, dport=dport, flags="S", seq=200), 2.0),
            stamp(Ether() / IP(src=server, dst=client) / TCP(sport=dport, dport=sport, flags="SA", seq=700, ack=201), 2.1),
            stamp(Ether() / IP(src=client, dst=server) / TCP(sport=sport, dport=dport, flags="A", seq=201, ack=701), 2.2),
            stamp(Ether() / IP(src=client, dst=server) / TCP(sport=sport, dport=dport, flags="PA", seq=201, ack=701) / Raw(b"second"), 2.3),
            stamp(Ether() / IP(src=client, dst=server) / UDP(sport=53000, dport=53) / Raw(b"one"), 3.0),
            stamp(Ether() / IP(src=server, dst=client) / UDP(sport=53, dport=53000) / Raw(b"two"), 10.0),
            stamp(Ether() / IP(src=client, dst=server) / UDP(sport=53000, dport=53) / Raw(b"three"), 41.0),
        ]
        wrpcap(path, packets)

    def test_real_pcap_is_split_consistently_for_pretrain_and_finetune(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pcap_path = os.path.join(temp_dir, "sessions.pcap")
            self.build_pcap(pcap_path)

            common_args = dict(
                pcap_path=pcap_path,
                token_bytes=2,
                cut_link_layer=True,
                udp_idle_timeout=30,
            )
            pretrain_sessions, pretrain_stats = collect_pretrain_flows(**common_args)
            finetune_sessions, finetune_stats = collect_finetune_flows(**common_args)

            self.assertEqual(len(pretrain_sessions), 4)
            self.assertEqual(len(finetune_sessions), 4)
            self.assertEqual(pretrain_stats["valid_packet_count"], 13)
            self.assertEqual(finetune_stats["valid_packet_count"], 13)

            tcp_sessions = [item for item in pretrain_sessions if item.protocol == 6]
            udp_sessions = [item for item in pretrain_sessions if item.protocol == 17]
            self.assertEqual([item.session_index for item in tcp_sessions], [1, 2])
            self.assertEqual([len(item.packets) for item in tcp_sessions], [6, 4])
            self.assertEqual([item.session_index for item in udp_sessions], [1, 2])
            self.assertEqual([len(item.packets) for item in udp_sessions], [2, 1])

            self.assertNotEqual(
                pretrain_sessions[0].packets[0]["tokens"],
                finetune_sessions[0].packets[0]["tokens"],
            )

            jsonl_path = os.path.join(temp_dir, "sessions.jsonl")
            write_stats = write_pcap_flows_to_part(
                flows=pretrain_sessions,
                pcap_path=pcap_path,
                pcap_index=1,
                part_path=jsonl_path,
                args=SimpleNamespace(
                    max_burst_num=24,
                    max_burst_len=128,
                    min_burst_len=1,
                    min_burst_num=1,
                    label_mode="unknown",
                    drop_unmatched_csv=False,
                ),
                label_map=None,
            )
            with open(jsonl_path, "r", encoding="utf-8") as reader:
                samples = [json.loads(line) for line in reader if line.strip()]

            self.assertEqual(write_stats["written_flow_count"], 4)
            self.assertEqual(len(samples), 4)
            self.assertEqual(
                [sample["session_index"] for sample in samples],
                [1, 2, 1, 2],
            )
            self.assertEqual(samples[0]["close_reason"], "tcp_rst")


if __name__ == "__main__":
    unittest.main()
