# main file for control the Tello with YOLOv5
# | IMPORT
from re import L
import time
import cv2
import numpy as np
import os
import sys

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

test = False
if __name__ == "__main__":
    if "-t" in sys.argv[1:] or "--test" in sys.argv[1:]:
        test = True
    cmd = Command(fmt=FORMAT, tello_ip=TELLO_IP, cmd_port=CMD_PORT, self_ip=IP)
    vh = VideoHandle(img_port=IMG_PORT, req_fps=FPS, test=test)
    vh.start_thrd()
    sent = cmd.send(msg="battery?")
    while vh.run:
        if vh.cur_cen is not None:
            cls_name, center, area, zone = vh.cur_cen
        time.sleep(0.5)
    cmd.term = True
