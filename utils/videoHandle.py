# handle video from YOLOv5
# | IMPORT
import cv2
import numpy as np
import sys
import threading
import time

from os import path
from typing import Tuple

from yolo.detect import detect, colors

# | CLASSES
class ML(detect):
    def __init__(self, source, size, conf_thres, weights, gray, gst) -> None:
        super().__init__()
        self.size = size
        self.source = source
        self.conf_thres = conf_thres
        self.weights = weights
        self.gray = gray
        self.gst = gst
        self.frame_color = None
        self.frame_gray = None

    def start(self):
        return super().start(
            weights=self.weights,
            source=self.source,
            imgsz=self.size,
            conf_thres=self.conf_thres,
            iou_thres=0.45,
            max_det=1000,
            device="",
            view_img=False,
            classes=None,
            gray=self.gray,
            raw=True,
            gst=self.gst,
        )

    def term(self):
        while self.alive():
            self.kill()
            time.sleep(0.5)


class VideoHandle:
    def __init__(self, img_port: int, req_fps: int):
        self.cur_objs = []
        self.obj_ids = {}
        self._id = 0
        self.count = 0

        self.frame_count = 0
        self.fps = []

        pipeline = f"udpsrc port={img_port} ! video/x-h264,stream-format=byte-stream,skip-first-bytes=2,framerate={req_fps}/1 ! queue ! decodebin ! videoconvert ! appsink"
        self.cur_path = sys.argv[0]
        self.det = ML(
            source=pipeline,
            size=[640, 640],
            conf_thres=0.6,
            weights=path.join(path.split(self.cur_path)[0], "yolo/mask_yolov5_gray.pt"),
            gray=True,
            gst=True,
        )

        self.frame_lst = []
        self.start = None
        self.cur_cen = None
        self.run = False

    def start_thrd(self):
        self.thrd = threading.Thread(target=self.__run)
        self.thrd.start()

    def __run(self):
        self.run = True
        self.det.start()
        while self.det.alive():
            gray_frame, color_frame, obj_lst_dct = self.det.get()

            if obj_lst_dct is not None:
                self.cur_objs = obj_lst_dct.copy()
                self.count += 1

            if self.count == 1:
                for obj in self.cur_objs:
                    self.obj_ids[self._id] = {"obj": obj, "lost": 0}
                    self._id += 1
            else:
                tmp_objs = self.cur_objs.copy()
                for i, old_obj in self.obj_ids.copy().items():
                    nearest_obj = None
                    min_dist = float("inf")

                    for cur_obj in tmp_objs:
                        dist = self.objs_dis(old_obj, cur_obj)
                        if dist < min_dist and dist < 100:
                            nearest_obj = cur_obj

                    if nearest_obj is None:
                        self.obj_ids[i]["lost"] += 1
                        if self.obj_ids[i]["lost"] >= 5 * (
                            1 / sum(self.fps) / len(self.fps)
                        ):
                            self.obj_ids.pop(i)
                    else:
                        self.obj_ids[i] = {"obj": nearest_obj, "lost": 0}
                        tmp_objs.remove(nearest_obj)

                for obj in tmp_objs:
                    self.obj_ids[self._id] = {"obj": obj, "lost": 0}
                    self._id += 1

            if gray_frame is not None and color_frame is not None:
                if self.start is None:
                    self.start = time.perf_counter()

                gray_draw_frame, cen = self.draw(gray_frame.copy())
                color_draw_frame, cen = self.draw(color_frame.copy())

                self.cur_cen = cen

                cv2.imshow("gray_frame", gray_draw_frame)
                cv2.imshow("color_frame", color_draw_frame)
                self.frame_lst.append(color_draw_frame)

                self.frame_count += 1
                time_del = time.perf_counter() - self.start
                self.fps.append(self.frame_count / time_del)

                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            time.sleep(1 / 60)

        cv2.destroyAllWindows()
        self.det.term()
        self.run = False

        avg_fps = sum(self.fps) / len(self.fps)
        self.write_vid(avg_fps)

    @staticmethod
    def objs_dis(o1, o2):
        cen1 = o1["obj"]["center"]
        cen2 = o2["center"]
        return np.linalg.norm(np.array(cen1) - np.array(cen2))

    @staticmethod
    def area_cal(tl: Tuple[int, int], br: Tuple[int, int]) -> int:
        w = br[0] - tl[0]
        h = tl[1] - br[1]
        return w * h

    def draw(self, img):
        cens = []
        for _id, obj in self.obj_ids.items():
            lost = obj["lost"]
            if lost > 0:
                continue

            obj_dct = obj["obj"]
            tl = obj_dct["topLeft"]
            br = obj_dct["bottomRight"]
            cen = obj_dct["center"]
            conf = obj_dct["confidenceScore"]
            name = obj_dct["className"]
            clss = obj_dct["classNumber"]
            label = f"{name} {conf:.2f}"
            id_label = f"id: {_id}"

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
            w, h = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                img,
                (br[0] - w - 10 - 10, br[1] - h - 10 - 10),
                (br[0], br[1]),
                colors(clss, True),
                -1,
            )
            cv2.putText(
                img,
                id_label,
                (br[0] - w - 10, br[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
            cv2.circle(img, cen, 3, colors(clss, True), -1)

            cens.append((cen, name, self.area_cal(tl, br)))

        return img, cens

    def write_vid(
        self, avg_fps: float, file_name: str = "out.mp4", fourcc: str = "mp4v"
    ):
        height, width, channels = self.frame_lst[0].shape
        file_name = path.join(path.split(self.cur_path)[0], file_name)
        writer = cv2.VideoWriter(
            file_name, cv2.VideoWriter_fourcc(*fourcc), avg_fps, (width, height)
        )
        for frame in self.frame_lst:
            writer.write(frame)
        writer.release()
