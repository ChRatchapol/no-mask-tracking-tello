# commanding module
# | IMPORT
import socket
import threading
import time

from typing import Tuple

# | CLASSES


class Command:
    def __init__(self, fmt="utf-8", tello_ip="192.168.10.1", cmd_port=8889, self_ip="") -> None:
        self.ip = self_ip
        self.gcs_cmd_addr = (self.ip, 9000)
        self.cmd_sock = self.__cmd_sock_setup()
        self.format = fmt
        self.tello_ip = tello_ip
        self.cmd_port = cmd_port
        self.run = False
        self.term = False

    def start(self):
        thrd = threading.Thread(target=self.recv_thrd)
        thrd.start()
        self.run = True
        self.term = False

    def recv_thrd(self):
        while not self.term:
            try:
                data, server = self.cmd_sock.recvfrom(1518)
            except BlockingIOError:
                continue
            else:
                print(data.decode(encoding=self.format).strip())
            time.sleep(1)

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
