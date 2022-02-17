# handle video from YOLOv5
# | IMPORT
import cv2
import numpy as np
import sys
import threading
import time

from os import path
from typing import Tuple

from yolo.detect import detect

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
    def __init__(self, img_port: int, req_fps: int, test: bool = False):
        self.cur_objs = []
        self.obj_ids = {}
        self._id = 0
        self.count = 0

        self.frame_count = 0
        self.fps = []

        if not test:
            pipeline = f"udpsrc port={img_port} ! video/x-h264,stream-format=byte-stream,skip-first-bytes=2,framerate={req_fps}/1 ! queue ! decodebin ! videoconvert ! appsink"
            self.cur_path = sys.argv[0]
            self.det = ML(
                source=pipeline,
                size=[640, 640],
                conf_thres=0.6,
                weights=path.join(
                    path.split(self.cur_path)[0], "yolo/mask_yolov5_gray.pt"
                ),
                gray=True,
                gst=True,
            )
        else:
            pipeline = "ksvideosrc device-index=0 ! decodebin ! videoconvert ! appsink"
            self.cur_path = sys.argv[0]
            self.det = ML(
                source=pipeline,
                size=[640, 640],
                conf_thres=0.6,
                weights=path.join(
                    path.split(self.cur_path)[0], "yolo/mask_yolov5_gray.pt"
                ),
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
        w = abs(br[0] - tl[0])
        h = abs(br[1] - tl[1])
        return w * h

    def in_zone(self, point, zone):
        if point is None:
            return False

        x_point, y_point = point
        tl, br = zone
        x_tl, y_tl = tl
        x_br, y_br = br

        if (x_point >= x_tl and x_point <= x_br) and (
            y_point >= y_tl and y_point <= y_br
        ):
            return True
        else:
            return False

    def draw_zone(self, img, center, factor=3):
        h, w, ch = img.shape
        tl = ((0, 0), (w // factor, h // factor))
        tc = ((w // factor, 0), (w - (w // factor), h // factor))
        tr = ((w - (w // factor), 0), (w, h // factor))

        ml = ((0, h // factor), (w // factor, h - (h // factor)))
        mc = ((w // factor, h // factor), (w - (w // factor), h - (h // factor)))
        mr = ((w - (w // factor), h // factor), (w, h - (h // factor)))

        bl = ((0, h - (h // factor)), (w // factor, h))
        bc = ((w // factor, h - (h // factor)), (w - (w // factor), h))
        br = ((w - (w // factor), h - (h // factor)), (w, h))

        zone_lst = [tl, tc, tr, ml, mc, mr, bl, bc, br]
        zone_res = None
        for i, z in enumerate(zone_lst):
            if self.in_zone(center, z):
                _z = zone_lst.pop(i)
                zone_lst.append(_z)
                zone_res = _z
                break

        for z in zone_lst:
            cv2.rectangle(
                img,
                (int(z[0][0]), int(z[0][1])),
                (int(z[1][0]), int(z[1][1])),
                (0, 255, 0) if self.in_zone(center, z) else (255, 255, 255),
                2,
            )
        return img, [n for n, v in locals().items() if v is zone_res][0]

    def draw(self, img):
        biggest_area = -1
        biggest_obj = None
        for _id, obj in self.obj_ids.items():
            lost = obj["lost"]
            if lost > 0:
                continue

            obj_dct = obj["obj"]
            tl = obj_dct["topLeft"]
            br = obj_dct["bottomRight"]

            if self.area_cal(tl, br) > biggest_area:
                biggest_area = self.area_cal(tl, br)
                biggest_obj = obj

        if biggest_obj is None:
            cen = None
            img, z_name = self.draw_zone(img, cen, factor=2.85)
            return img, None
        else:
            obj_dct = biggest_obj["obj"]
            tl = obj_dct["topLeft"]
            br = obj_dct["bottomRight"]
            cen = obj_dct["center"]
            conf = obj_dct["confidenceScore"]
            name = obj_dct["className"]
            clss = obj_dct["classNumber"]
            label = f"{name} {conf:.2f} {self.area_cal(tl, br)}"
            id_label = f"id: {_id}"

            colors = [(157, 159, 21), (62, 139, 2), (67, 74, 250)]

            img, z_name = self.draw_zone(img, cen, factor=2.85)

            cv2.rectangle(img, tl, br, colors[clss], 3)
            w, h = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
            cv2.rectangle(
                img,
                (tl[0] - 2, tl[1] - h - 10 - 10),
                (tl[0] + w + 2, tl[1]),
                colors[clss],
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
                colors[clss],
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
            cv2.circle(img, cen, 3, colors[clss], -1)
            return img, (name, cen, self.area_cal(tl, br), z_name)

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
