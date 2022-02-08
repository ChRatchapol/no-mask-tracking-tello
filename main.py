# main file for control the Tello with YOLOv5
# | IMPORT
import cv2
import numpy as np
import os

from utils.command import Command

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

if __name__ == "__main__":
    cmd = Command(fmt=FORMAT, tello_ip=TELLO_IP, cmd_port=CMD_PORT, self_ip=IP)
    cmd.init()
