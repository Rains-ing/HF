import sys
import types
import unittest


try:
    import scapy.all  # noqa: F401
except ModuleNotFoundError:
    fake_scapy = types.ModuleType("scapy")
    fake_scapy_all = types.ModuleType("scapy.all")

    class IP:
        pass

    class TCP:
        pass

    class UDP:
        pass

    fake_scapy_all.IP = IP
    fake_scapy_all.TCP = TCP
    fake_scapy_all.UDP = UDP
    fake_scapy_all.PcapReader = object
    fake_scapy.all = fake_scapy_all
    sys.modules["scapy"] = fake_scapy
    sys.modules["scapy.all"] = fake_scapy_all


from process_pretrain_data.flow_sessionizer import (  # noqa: E402
    FlowSessionizer,
    IP,
    TCP,
    UDP,
)


class FakeIPLayer:
    def __init__(self, src, dst, proto):
        self.src = src
        self.dst = dst
        self.proto = proto


class FakeTransportLayer:
    def __init__(self, sport, dport, flags=0, seq=0, ack=0, payload=b""):
        self.sport = sport
        self.dport = dport
        self.flags = flags
        self.seq = seq
        self.ack = ack
        self.payload = payload


class FakePacket:
    def __init__(self, timestamp, layers):
        self.time = timestamp
        self.layers = layers

    def __contains__(self, layer):
        return layer in self.layers

    def __getitem__(self, layer):
        return self.layers[layer]


def tcp_packet(timestamp, reverse=False, flags=0x10, seq=0, ack=0, payload=b""):
    if reverse:
        src, sport, dst, dport = "10.0.0.2", 443, "10.0.0.1", 50000
    else:
        src, sport, dst, dport = "10.0.0.1", 50000, "10.0.0.2", 443
    return FakePacket(
        timestamp,
        {
            IP: FakeIPLayer(src, dst, 6),
            TCP: FakeTransportLayer(sport, dport, flags, seq, ack, payload),
        },
    )


def udp_packet(timestamp, reverse=False):
    if reverse:
        src, sport, dst, dport = "10.0.0.2", 53, "10.0.0.1", 53000
    else:
        src, sport, dst, dport = "10.0.0.1", 53000, "10.0.0.2", 53
    return FakePacket(
        timestamp,
        {
            IP: FakeIPLayer(src, dst, 17),
            UDP: FakeTransportLayer(sport, dport),
        },
    )


class FlowSessionizerTest(unittest.TestCase):
    def make_sessionizer(self):
        return FlowSessionizer(
            packet_to_tokens=lambda pkt: ["abcd"],
            tcp_idle_timeout=120,
            tcp_handshake_timeout=30,
            tcp_close_timeout=15,
            udp_idle_timeout=30,
        )

    def test_syn_retransmission_stays_in_same_session_but_new_syn_splits(self):
        sessionizer = self.make_sessionizer()
        packets = [
            tcp_packet(0.0, flags=0x02, seq=100),
            tcp_packet(0.1, flags=0x02, seq=100),
            tcp_packet(0.2, reverse=True, flags=0x12, seq=500, ack=101),
            tcp_packet(0.25, flags=0x02, seq=100),
            tcp_packet(0.3, flags=0x10, seq=101, ack=501),
            tcp_packet(1.0, flags=0x18, seq=101, ack=501, payload=b"one"),
            tcp_packet(1.1, reverse=True, flags=0x14, seq=501, ack=104),
            tcp_packet(2.0, flags=0x02, seq=200),
            tcp_packet(2.1, reverse=True, flags=0x12, seq=700, ack=201),
            tcp_packet(2.2, flags=0x10, seq=201, ack=701),
        ]
        for packet in packets:
            sessionizer.feed(packet)

        sessions = sessionizer.finish_all()
        self.assertEqual([item.session_index for item in sessions], [1, 2])
        self.assertEqual([len(item.packets) for item in sessions], [7, 3])
        self.assertEqual(sessions[0].close_reason, "tcp_rst")
        self.assertEqual(sessions[1].close_reason, "pcap_end")
        self.assertEqual(sessions[0].packets[0]["dir"], 0)
        self.assertEqual(sessions[0].packets[2]["dir"], 1)

    def test_packets_after_rst_do_not_form_a_spurious_session(self):
        sessionizer = self.make_sessionizer()
        for packet in [
            tcp_packet(0.0, flags=0x02, seq=100),
            tcp_packet(0.1, reverse=True, flags=0x12, seq=500, ack=101),
            tcp_packet(0.2, flags=0x10, seq=101, ack=501),
            tcp_packet(1.0, reverse=True, flags=0x14, seq=501, ack=101),
            tcp_packet(1.1, flags=0x10, seq=101, ack=502),
        ]:
            sessionizer.feed(packet)

        sessions = sessionizer.finish_all()
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].close_reason, "tcp_rst")
        self.assertEqual(len(sessions[0].packets), 5)

    def test_complete_fin_handshake_closes_session(self):
        sessionizer = self.make_sessionizer()
        packets = [
            tcp_packet(0.0, flags=0x02, seq=100),
            tcp_packet(0.1, reverse=True, flags=0x12, seq=500, ack=101),
            tcp_packet(0.2, flags=0x10, seq=101, ack=501),
            tcp_packet(1.0, flags=0x11, seq=101, ack=501),
            tcp_packet(1.1, reverse=True, flags=0x10, seq=501, ack=102),
            tcp_packet(1.2, reverse=True, flags=0x11, seq=501, ack=102),
            tcp_packet(1.3, flags=0x10, seq=102, ack=502),
        ]
        for packet in packets:
            sessionizer.feed(packet)

        sessions = sessionizer.finish_all()
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].close_reason, "tcp_fin")
        self.assertEqual(len(sessions[0].packets), 7)

    def test_udp_idle_timeout_creates_new_session(self):
        sessionizer = self.make_sessionizer()
        for packet in [
            udp_packet(0.0),
            udp_packet(10.0, reverse=True),
            udp_packet(41.0),
        ]:
            sessionizer.feed(packet)

        sessions = sessionizer.finish_all()
        self.assertEqual([item.session_index for item in sessions], [1, 2])
        self.assertEqual([len(item.packets) for item in sessions], [2, 1])
        self.assertEqual(sessions[0].close_reason, "udp_idle_timeout")


if __name__ == "__main__":
    unittest.main()
