# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# from utils.datasets import LoadStreams
from models.common import DetectMultiBackend

from yolo.utils.general import (
    check_img_size,
    check_imshow,
    increment_path,
    non_max_suppression,
    print_args,
    scale_coords,
    strip_optimizer,
)
from yolo.utils.plots import colors
from yolo.utils.torch_utils import select_device, time_sync

from threading import Thread

from yolo.utils.general import LOGGER


class LoadCam:
    def __init__(
        self,
        source="streams.txt",
        img_size=640,
        stride=32,
        auto=True,
        gray=False,
        gst=False,
    ):
        self.mode = "stream"
        self.img_size = img_size
        self.stride = stride
        self.gray = gray
        self.term = False

        self.imgs1, self.imgs2, self.fps, self.frames, self.threads = (
            None,
            None,
            0,
            0,
            None,
        )
        # self.source = clean_str(source)  # clean source names for later
        self.source = source
        self.auto = auto
        # Start thread to read frames from video stream
        st = f"source: {source}... "

        source = (
            eval(source) if source.isnumeric() else source
        )  # i.e. s = '0' local webcam
        if gst:
            cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(source)

        assert cap.isOpened(), f"{st}Failed to open {source}"
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0  # 30 FPS fallback
        self.frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float(
            "inf"
        )  # infinite stream fallback

        _, tmp_im = cap.read()  # guarantee first frame
        if self.gray:
            tmp_gray = cv2.cvtColor(tmp_im, cv2.COLOR_BGR2GRAY)
            self.imgs1 = np.zeros_like(tmp_im)
            self.imgs1[:, :, 0] = tmp_gray
            self.imgs1[:, :, 1] = tmp_gray
            self.imgs1[:, :, 2] = tmp_gray
        else:
            self.imgs1 = tmp_im
        self.imgs2 = tmp_im

        self.threads = Thread(target=self.update, args=([cap, source]), daemon=True)
        self.threads.start()
        LOGGER.info(
            f"{st} Success ({self.frames} frames {w}x{h} at {self.fps:.2f} FPS)\n"
        )

    def update(self, cap, stream):
        n, f, read = (
            0,
            self.frames,
            1,
        )  # frame number, frame array, inference every 'read' frame
        timer = time.perf_counter() # control fps
        while cap.isOpened() and n < f:
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n % read == 0 and (time.perf_counter() - timer > (1/(self.fps*2))):
                timer = time.perf_counter()
                success, tmp_im = cap.retrieve()
                tmp_im = cv2.resize(tmp_im, (640, 480), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
                if self.gray:
                    tmp_gray = cv2.cvtColor(tmp_im, cv2.COLOR_BGR2GRAY)
                    im = np.zeros_like(tmp_im)
                    im[:, :, 0] = tmp_gray
                    im[:, :, 1] = tmp_gray
                    im[:, :, 2] = tmp_gray
                else:
                    im = tmp_im

                if success:
                    self.imgs1 = im
                    self.imgs2 = tmp_im
                else:
                    LOGGER.warning(
                        "WARNING: Video stream unresponsive, please check your IP camera connection."
                    )
                    self.imgs1 = np.zeros_like(self.imgs1)
                    self.imgs2 = np.zeros_like(self.imgs2)
                    cap.open(stream)  # re-open stream if signal was lost

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if (
            not self.threads.is_alive() or cv2.waitKey(1) == ord("q") or self.term
        ):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # img0 = self.imgs1.copy()
        # img1 = self.imgs2.copy()
        img = self.imgs2.copy()

        # Convert
        img = img[..., ::-1].transpose((2, 0, 1))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return self.source, img, self.imgs1, self.imgs2, None, ""


class detect:
    def __init__(self) -> None:
        self.obj_lst = []
        self.frame0 = None
        self.frame1 = None
        self.thrd = None
        self.dataset = None
        self.names = None

    def start(
        self,
        weights=ROOT / "yolov5x6.pt",  # model.pt path(s)
        source="0",  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / "runs/detect",  # save results to project/name
        name="exp",  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        gray=False,  # receive video as gray scale):
        raw=False,  # get raw frame and obj_lst insteed
        gst=False,  # use gstreamer
    ):
        self.thrd = Thread(
            target=self.run,
            args=(
                weights,
                source,
                imgsz,
                conf_thres,
                iou_thres,
                max_det,
                device,
                view_img,
                classes,
                agnostic_nms,
                augment,
                visualize,
                update,
                project,
                name,
                exist_ok,
                half,
                dnn,
                gray,
                raw,
                gst,
            ),
        )
        self.thrd.start()

    def alive(self):
        return self.thrd.is_alive()

    def kill(self):
        if self.dataset is not None:
            self.dataset.term = True

    @torch.no_grad()
    def run(
        self,
        weights=ROOT / "yolov5x6.pt",  # model.pt path(s)
        source="0",  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / "runs/detect",  # save results to project/name
        name="exp",  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        gray=False,  # receive video as gray scale):
        raw=False,  # get raw frame and obj_lst insteed
        gst=False,  # use gstreamer
    ):
        # Directories
        save_dir = increment_path(
            Path(project) / name, exist_ok=exist_ok
        )  # increment run

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, self.names, pt, jit, onnx, engine = (
            model.stride,
            model.names,
            model.pt,
            model.jit,
            model.onnx,
            model.engine,
        )
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Half
        half &= (
            pt or jit or engine
        ) and device.type != "cpu"  # half precision only supported by PyTorch on CUDA
        if pt or jit:
            model.model.half() if half else model.model.float()

        # Dataloader
        chk_imshow = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.dataset = LoadCam(
            source, img_size=imgsz, stride=stride, auto=pt, gray=gray, gst=gst
        )
        # bs = len(self.dataset)  # batch_size

        # Run inference
        model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, im1s, vid_cap, s in self.dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            visualize = (
                increment_path(save_dir / Path(path).stem, mkdir=True)
                if visualize
                else False
            )
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(
                pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
            )
            dt[2] += time_sync() - t3

            # Process predictions
            self.obj_lst = []
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, im1, frame = (
                    path[i],
                    im0s.copy(),
                    im1s.copy(),
                    self.dataset.count,
                )
                self.res = im0
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        im.shape[2:], det[:, :4], im0.shape
                    ).round()

                    self.obj_lst = [list(map(float, i)) for i in det]

                # Stream results
                if not raw:
                    im0 = self.__draw(im0)
                    im1 = self.__draw(im1)

                self.frame0 = im0
                self.frame1 = im1
                if view_img and chk_imshow:
                    cv2.imshow("frame0", im0)
                    cv2.imshow("frame1", im1)
                    cv2.waitKey(1)  # 1 millisecond

        if update:
            strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    def __draw(self, img):
        obj_lst_dct = self.obj_lst2obj_lst_dct(self.obj_lst, self.names)

        for obj_dct in obj_lst_dct:
            tl = obj_dct["topLeft"]
            br = obj_dct["bottomRight"]
            cen = obj_dct["center"]
            conf = obj_dct["confidenceScore"]
            name = obj_dct["className"]
            clss = obj_dct["classNumber"]
            label = f"{name} {conf:.2f}"

            cv2.rectangle(img, tl, br, colors(clss, True), 3)
            w, h = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
            cv2.rectangle(
                img,
                (tl[0] - 2, tl[1] - h - 10 - 10),
                (tl[0] + w + 2, tl[1]),
                colors(clss, True),
                -1,
            )
            cv2.putText(
                img,
                label,
                (tl[0], tl[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )
            cv2.circle(img, cen, 3, colors(clss, True), -1)

        return img

    @staticmethod
    def obj_lst2obj_lst_dct(obj_lst, names):
        if names is None:
            return None

        res = []
        for obj in obj_lst:
            tl = tuple(map(int, obj[:2]))
            br = tuple(map(int, obj[2:4]))
            cen = (
                tl[0] + ((br[0] - tl[0]) // 2),
                (tl[1] + ((br[1] - tl[1]) // 2)),
            )
            conf = float(obj[4])
            c = names[int(obj[5])]
            res.append(
                {
                    "topLeft": tl,
                    "bottomRight": br,
                    "center": cen,
                    "confidenceScore": conf,
                    "className": c,
                    "classNumber": int(obj[5]),
                }
            )

        return res

    def get(self):
        return (
            self.frame0,
            self.frame1,
            self.obj_lst2obj_lst_dct(self.obj_lst, self.names),
        )


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default=ROOT / "yolov5x6.pt",
        help="model path(s)",
    )
    parser.add_argument(
        "--source", type=str, default="0", help="file/dir/URL/glob, 0 for webcam"
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[640],
        help="inference size h,w",
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="NMS IoU threshold"
    )
    parser.add_argument(
        "--max-det", type=int, default=1000, help="maximum detections per image"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --classes 0, or --classes 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument(
        "--project", default=ROOT / "runs/detect", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument(
        "--half", action="store_true", help="use FP16 half-precision inference"
    )
    parser.add_argument(
        "--dnn", action="store_true", help="use OpenCV DNN for ONNX inference"
    )
    parser.add_argument(
        "--gray", action="store_true", help="receive video as gray scale"
    )
    parser.add_argument(
        "--raw", action="store_true", help="get raw frame and obj_lst insteed"
    )
    parser.add_argument("--gst", action="store_true", help="use gstreamer")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def draw(img, obj_lst_dct):
    for obj_dct in obj_lst_dct:
        tl = obj_dct["topLeft"]
        br = obj_dct["bottomRight"]
        cen = obj_dct["center"]
        conf = obj_dct["confidenceScore"]
        name = obj_dct["className"]
        clss = obj_dct["classNumber"]
        label = f"{name} {conf:.2f}"

        cv2.rectangle(img, tl, br, colors(clss, True), 3)
        w, h = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
        cv2.rectangle(
            img,
            (tl[0] - 2, tl[1] - h - 10 - 10),
            (tl[0] + w + 2, tl[1]),
            colors(clss, True),
            -1,
        )
        cv2.putText(
            img,
            label,
            (tl[0], tl[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
        cv2.circle(img, cen, 3, colors(clss, True), -1)

    return img


def main(opt):
    det = detect()
    det.start(**vars(opt))

    while det.alive():
        frame, obj_lst_dct = det.get()

        if frame is not None:
            cv2.imshow("Output", frame)
            if opt.raw:
                cv2.imshow("Draw", draw(frame.copy(), obj_lst_dct))
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cv2.destroyAllWindows()
    while det.alive():
        det.kill()
        time.sleep(0.5)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
