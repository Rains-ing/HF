from dataclasses import dataclass, field
import heapq

from scapy.all import PcapReader, IP, TCP, UDP


TCP_PROTOCOL = 6
UDP_PROTOCOL = 17
SEQ_MODULO = 1 << 32
SEQ_HALF = 1 << 31


def _endpoint(src_ip, src_port):
    return str(src_ip), int(src_port)


def _normalize_protocol(proto):
    return int(proto)


def make_biflow_key(src_ip, src_port, dst_ip, dst_port, proto):
    """Return a direction-independent IPv4 TCP/UDP five-tuple."""
    endpoint_a = _endpoint(src_ip, src_port)
    endpoint_b = _endpoint(dst_ip, dst_port)
    proto = _normalize_protocol(proto)

    if endpoint_a <= endpoint_b:
        return endpoint_a[0], endpoint_a[1], endpoint_b[0], endpoint_b[1], proto
    return endpoint_b[0], endpoint_b[1], endpoint_a[0], endpoint_a[1], proto


def _seq_at_or_after(value, target):
    """TCP serial-number comparison with wraparound support."""
    return ((int(value) - int(target)) % SEQ_MODULO) < SEQ_HALF


@dataclass
class FlowSession:
    flow_key: tuple
    session_index: int
    start_time: float
    end_time: float
    initiator: tuple
    midstream: bool
    order: int
    packets: list = field(default_factory=list)
    close_reason: str = "pcap_end"
    last_seen: float = 0.0
    deadline: float = 0.0
    deadline_version: int = 0
    tcp_state: str = "midstream"
    terminal_reason: str | None = None
    initial_syn_seq: int | None = None
    initial_syn_direction: int | None = None
    fin_end_seq: dict = field(default_factory=dict)
    fin_acked: set = field(default_factory=set)

    @property
    def protocol(self):
        return int(self.flow_key[4])

    @property
    def duration(self):
        return max(0.0, float(self.end_time) - float(self.start_time))

    def packet_direction(self, src_ip, src_port):
        return 0 if _endpoint(src_ip, src_port) == self.initiator else 1


class FlowSessionizer:
    """
    Split one PCAP into bidirectional TCP/UDP sessions.

    TCP boundaries use SYN generations, RST, the complete two-sided FIN/ACK
    close, and protocol-specific fallback timeouts. UDP uses an inactivity
    timeout because UDP has no connection-close signal.
    """

    def __init__(
        self,
        packet_to_tokens,
        max_packets_per_flow=None,
        tcp_idle_timeout=120.0,
        tcp_handshake_timeout=30.0,
        tcp_close_timeout=15.0,
        udp_idle_timeout=30.0,
    ):
        self.packet_to_tokens = packet_to_tokens
        self.max_packets_per_flow = max_packets_per_flow
        self.tcp_idle_timeout = self._positive_timeout(tcp_idle_timeout)
        self.tcp_handshake_timeout = self._positive_timeout(tcp_handshake_timeout)
        self.tcp_close_timeout = self._positive_timeout(tcp_close_timeout)
        self.udp_idle_timeout = self._positive_timeout(udp_idle_timeout)

        self.active = {}
        self.completed = []
        self.session_counts = {}
        self.deadlines = []
        self.next_order = 0

        self.packet_count = 0
        self.valid_packet_count = 0
        self.skipped_packet_count = 0
        self.truncated_packet_count = 0
        self.out_of_order_packet_count = 0
        self._last_capture_time = None

    @staticmethod
    def _positive_timeout(value):
        if value is None:
            return None
        value = float(value)
        return value if value > 0 else None

    @staticmethod
    def _packet_tuple(pkt):
        if IP not in pkt:
            return None

        ip = pkt[IP]
        if TCP in pkt:
            return ip.src, int(pkt[TCP].sport), ip.dst, int(pkt[TCP].dport), TCP_PROTOCOL
        if UDP in pkt:
            return ip.src, int(pkt[UDP].sport), ip.dst, int(pkt[UDP].dport), UDP_PROTOCOL
        return None

    def _timeout_for(self, session):
        if session.protocol == UDP_PROTOCOL:
            return self.udp_idle_timeout
        if session.tcp_state in {"syn_sent", "syn_ack_seen"}:
            return self.tcp_handshake_timeout
        if session.tcp_state in {"closing", "closed", "reset"}:
            return self.tcp_close_timeout
        return self.tcp_idle_timeout

    def _schedule_deadline(self, session):
        timeout = self._timeout_for(session)
        session.deadline_version += 1
        if timeout is None:
            session.deadline = 0.0
            return

        session.deadline = session.last_seen + timeout
        heapq.heappush(
            self.deadlines,
            (
                session.deadline,
                session.deadline_version,
                session.flow_key,
                session.session_index,
            ),
        )

    def _expire_before(self, timestamp):
        while self.deadlines and self.deadlines[0][0] < timestamp:
            deadline, version, flow_key, session_index = heapq.heappop(self.deadlines)
            session = self.active.get(flow_key)
            if session is None:
                continue
            if session.session_index != session_index:
                continue
            if session.deadline_version != version or session.deadline != deadline:
                continue

            if session.terminal_reason is not None:
                reason = session.terminal_reason
            elif session.protocol == UDP_PROTOCOL:
                reason = "udp_idle_timeout"
            elif session.tcp_state in {"syn_sent", "syn_ack_seen"}:
                reason = "tcp_handshake_timeout"
            elif session.tcp_state == "closing":
                reason = "tcp_close_timeout"
            else:
                reason = "tcp_idle_timeout"
            self._finish(session, reason)

    def _new_session(self, flow_key, timestamp, initiator, midstream, tcp_state):
        session_index = self.session_counts.get(flow_key, 0) + 1
        self.session_counts[flow_key] = session_index
        self.next_order += 1

        session = FlowSession(
            flow_key=flow_key,
            session_index=session_index,
            start_time=timestamp,
            end_time=timestamp,
            last_seen=timestamp,
            initiator=initiator,
            midstream=midstream,
            order=self.next_order,
            tcp_state=tcp_state,
        )
        self.active[flow_key] = session
        return session

    def _finish(self, session, reason):
        if self.active.get(session.flow_key) is not session:
            return
        session.close_reason = reason
        del self.active[session.flow_key]
        self.completed.append(session)

    @staticmethod
    def _tcp_payload_length(pkt):
        tcp = pkt[TCP]
        return len(bytes(tcp.payload))

    @staticmethod
    def _is_initial_syn(pkt):
        flags = int(pkt[TCP].flags)
        return bool(flags & 0x02) and not bool(flags & 0x10)

    def _is_syn_retransmission(self, session, pkt, direction):
        if session.tcp_state not in {"syn_sent", "syn_ack_seen"}:
            return False
        if session.initial_syn_seq is None:
            return False
        return (
            direction == session.initial_syn_direction
            and int(pkt[TCP].seq) == session.initial_syn_seq
        )

    def _update_tcp_state(self, session, pkt, direction):
        tcp = pkt[TCP]
        flags = int(tcp.flags)
        has_syn = bool(flags & 0x02)
        has_ack = bool(flags & 0x10)
        has_fin = bool(flags & 0x01)

        if has_syn and not has_ack and session.initial_syn_seq is None:
            session.initial_syn_seq = int(tcp.seq)
            session.initial_syn_direction = direction
            session.tcp_state = "syn_sent"
        elif session.tcp_state == "syn_sent" and has_syn and has_ack:
            session.tcp_state = "syn_ack_seen"
        elif (
            session.tcp_state in {"syn_sent", "syn_ack_seen"}
            and has_ack
            and not has_syn
            and direction == session.initial_syn_direction
        ):
            session.tcp_state = "established"

        if has_ack:
            other_direction = 1 - direction
            other_fin_end = session.fin_end_seq.get(other_direction)
            if other_fin_end is not None and _seq_at_or_after(int(tcp.ack), other_fin_end):
                session.fin_acked.add(other_direction)

        if has_fin:
            fin_end = (
                int(tcp.seq)
                + self._tcp_payload_length(pkt)
                + 1
            ) % SEQ_MODULO
            session.fin_end_seq[direction] = fin_end
            session.tcp_state = "closing"

    @staticmethod
    def _clean_tcp_close_complete(session):
        return (
            0 in session.fin_end_seq
            and 1 in session.fin_end_seq
            and 0 in session.fin_acked
            and 1 in session.fin_acked
        )

    def _append_packet(self, session, pkt, direction, timestamp):
        session.last_seen = max(session.last_seen, timestamp)
        session.end_time = max(session.end_time, timestamp)

        if (
            self.max_packets_per_flow is not None
            and len(session.packets) >= int(self.max_packets_per_flow)
        ):
            self.truncated_packet_count += 1
            return

        tokens = self.packet_to_tokens(pkt)
        if not tokens:
            self.skipped_packet_count += 1
            return

        session.packets.append({
            "dir": direction,
            "tokens": tokens,
            "timestamp": timestamp,
        })
        self.valid_packet_count += 1

    def feed(self, pkt):
        self.packet_count += 1
        five_tuple = self._packet_tuple(pkt)
        if five_tuple is None:
            self.skipped_packet_count += 1
            return

        try:
            timestamp = float(pkt.time)
            src_ip, src_port, dst_ip, dst_port, proto = five_tuple
            flow_key = make_biflow_key(
                src_ip, src_port, dst_ip, dst_port, proto
            )
        except Exception:
            self.skipped_packet_count += 1
            return

        if self._last_capture_time is not None and timestamp < self._last_capture_time:
            self.out_of_order_packet_count += 1
        else:
            self._last_capture_time = timestamp
            self._expire_before(timestamp)

        session = self.active.get(flow_key)

        if proto == TCP_PROTOCOL:
            tcp = pkt[TCP]
            initial_syn = self._is_initial_syn(pkt)

            if session is not None:
                provisional_direction = session.packet_direction(src_ip, src_port)
                if initial_syn and not self._is_syn_retransmission(
                    session, pkt, provisional_direction
                ):
                    self._finish(
                        session,
                        session.terminal_reason or "new_syn",
                    )
                    session = None

            if session is None:
                if initial_syn:
                    initiator = _endpoint(src_ip, src_port)
                    midstream = False
                    tcp_state = "syn_sent"
                elif bool(int(tcp.flags) & 0x02) and bool(int(tcp.flags) & 0x10):
                    initiator = _endpoint(dst_ip, dst_port)
                    midstream = True
                    tcp_state = "midstream"
                else:
                    initiator = _endpoint(src_ip, src_port)
                    midstream = True
                    tcp_state = "midstream"
                session = self._new_session(
                    flow_key, timestamp, initiator, midstream, tcp_state
                )

            direction = session.packet_direction(src_ip, src_port)
            self._append_packet(session, pkt, direction, timestamp)
            self._update_tcp_state(session, pkt, direction)

            if bool(int(tcp.flags) & 0x04):
                session.tcp_state = "reset"
                if session.terminal_reason is None:
                    session.terminal_reason = "tcp_rst"
            elif self._clean_tcp_close_complete(session):
                session.tcp_state = "closed"
                session.terminal_reason = "tcp_fin"
            self._schedule_deadline(session)

        else:
            if session is None:
                session = self._new_session(
                    flow_key,
                    timestamp,
                    _endpoint(src_ip, src_port),
                    midstream=False,
                    tcp_state="udp",
                )

            direction = session.packet_direction(src_ip, src_port)
            self._append_packet(session, pkt, direction, timestamp)
            self._schedule_deadline(session)

    def finish_all(self):
        for session in sorted(self.active.values(), key=lambda item: item.order):
            self._finish(session, session.terminal_reason or "pcap_end")
        self.completed.sort(key=lambda item: item.order)
        return self.completed

    def stats(self):
        return {
            "packet_count": self.packet_count,
            "valid_packet_count": self.valid_packet_count,
            "skipped_packet_count": self.skipped_packet_count,
            "truncated_packet_count": self.truncated_packet_count,
            "out_of_order_packet_count": self.out_of_order_packet_count,
        }


def collect_sessions_from_pcap(
    pcap_path,
    packet_to_tokens,
    max_packets_per_flow=None,
    tcp_idle_timeout=120.0,
    tcp_handshake_timeout=30.0,
    tcp_close_timeout=15.0,
    udp_idle_timeout=30.0,
):
    sessionizer = FlowSessionizer(
        packet_to_tokens=packet_to_tokens,
        max_packets_per_flow=max_packets_per_flow,
        tcp_idle_timeout=tcp_idle_timeout,
        tcp_handshake_timeout=tcp_handshake_timeout,
        tcp_close_timeout=tcp_close_timeout,
        udp_idle_timeout=udp_idle_timeout,
    )

    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            sessionizer.feed(pkt)

    return sessionizer.finish_all(), sessionizer.stats()
