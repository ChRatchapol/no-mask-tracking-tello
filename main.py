# main file for control the Tello with YOLOv5
# | IMPORT
import time
import cv2
import numpy as np
import os
import sys

from queue import Empty

from utils.command import Command
from utils.videoHandle import VideoHandle

# | GLOBAL VARIABLES
FORMAT = "utf-8"
IP = ""
CMD_PORT = 8889
STT_PORT = 8890
IMG_PORT = 11111
TELLO_IP = "192.168.10.1"

FPS = 15

BLU = (0, 0, 255)
CYN = (0, 255, 255)
GRN = (0, 255, 0)
MGT = (255, 0, 255)
RED = (255, 0, 0)
YLE = (255, 255, 0)


# | Functions


def move_by_zone(zone: str, area: int) -> None:
    global cmd

    zone_dct = {
        "tl": "up-ccw",
        "tc": "up",
        "tr": "up-cw",
        "ml": "ccw",
        "mc": "",
        "mr": "cw",
        "bl": "down-ccw",
        "bc": "down",
        "br": "down-cw",
    }

    cmd_str = zone_dct[zone]

    if area < 7000:
        z_cmd = "forward"
    elif area > 12000:
        z_cmd = "back"
    else:
        z_cmd = ""

    if cmd_str == "":
        return
    else:
        cmd_lst = cmd_str.split("-")
        if z_cmd != "":
            cmd_lst.append(z_cmd)

    # print(cmd_lst)

    for _cmd in cmd_lst:
        if _cmd in ["ccw", "cw"]:
            amp = "15"
        elif _cmd in ["forward", "back"]:
            amp = "20"
        else:
            amp = "15"
        print(f"{_cmd} {amp}")
        sent = False
        for _ in range(20):
            if not sent:
                sent = cmd.send(msg=f"{_cmd} {amp}")
                sent = True
            time.sleep(0.1)

test = False
if __name__ == "__main__":
    if "-t" in sys.argv[1:] or "--test" in sys.argv[1:]:
        test = True
    if not test:
        cmd = Command(fmt=FORMAT, tello_ip=TELLO_IP, cmd_port=CMD_PORT, self_ip=IP)
        cmd.init()
    vh = VideoHandle(img_port=IMG_PORT, req_fps=FPS, test=test)
    vh.start_thrd()
    if not test:
        sent = cmd.send(msg="battery?")
        sent = cmd.send(msg="takeoff")
        time.sleep(3)
        while vh.run:
            try:
                response = cmd.rcv_queue.get_nowait()
            except Empty:
                pass
            else:
                print(">>> " + response)

            if vh.cur_cen is not None:
                cls_name, center, area, zone = vh.cur_cen
                if cls_name != "with_mask":
                    move_by_zone(zone, area)
                else:
                    sent = cmd.send(msg="battery?")
            time.sleep(0.5)
        sent = cmd.send(msg="land")
        cmd.rcv_term()
    else:
        while vh.run:
            time.sleep(0.5+0.15)
