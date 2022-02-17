# commanding module
# | IMPORT
import multiprocessing
import socket
import threading
import time

from typing import Tuple
from queue import Empty, Full

# | CLASSES


class Command:
    def __init__(
        self, fmt="utf-8", tello_ip="192.168.10.1", cmd_port=8889, self_ip=""
    ) -> None:
        self.ip = self_ip
        self.gcs_cmd_addr = (self.ip, 9000)
        self.cmd_sock = self.__cmd_sock_setup()
        self.format = fmt
        self.tello_ip = tello_ip
        self.cmd_port = cmd_port
        self.term = False

    def start(self):
        # thrd = threading.Thread(target=self.recv_proc)
        # thrd.start()
        self.rcv_queue = multiprocessing.Queue()
        self.cmd_queue = multiprocessing.Queue(1)
        self.proc = multiprocessing.Process(
            target=self.recv_proc,
            args=(self.cmd_sock, self.rcv_queue, self.cmd_queue, self.format),
        )
        self.proc.start()
        self.term = False

    def rcv_term(self):
        while self.proc.is_alive():
            try:
                self.cmd_queue.put_nowait("term")
            except Full:
                pass
            time.sleep(0.5)

    def recv_proc(
        self,
        cmd_sock: socket.socket,
        rcv_q: multiprocessing.Queue,
        cmd_q: multiprocessing.Queue,
        format: str = "utf-8",
    ):
        while True:
            try:
                cmd = cmd_q.get_nowait()
                if cmd == "term":
                    break
            except Empty:
                pass

            try:
                data, server = cmd_sock.recvfrom(1518)
            except BlockingIOError:
                continue
            else:
                rcv_q.put(data.decode(encoding=format).strip())
                # print(data.decode(encoding=self.format).strip())
            time.sleep(1)
        rcv_q.close()
        cmd_q.close()

    def __cmd_sock_setup(self) -> socket.socket:
        soc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        soc.bind(self.gcs_cmd_addr)
        soc.setblocking(False)
        return soc

    def send(self, msg: str, ip: str = None, port: int = None) -> int:
        if ip is None:
            ip = self.tello_ip
        if port is None:
            port = self.cmd_port

        msg = msg.encode(encoding=self.format)
        sent = self.cmd_sock.sendto(msg, (ip, port))
        return sent

    def init(self) -> Tuple[int, int]:
        self.start()
        tmp1 = self.send(msg="command", ip=self.tello_ip, port=self.cmd_port)
        tmp2 = self.send(msg="streamon", ip=self.tello_ip, port=self.cmd_port)
        return (tmp1, tmp2)
