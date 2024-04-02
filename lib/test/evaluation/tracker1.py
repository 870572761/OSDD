import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv
from matplotlib.widgets import Slider
from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *


def trackerlist(
    name: str,
    parameter_name: str,
    dataset_name: str,
    run_ids=None,
    display_name: str = None,
    result_only=False,
):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [
        Tracker1(name, parameter_name, dataset_name, run_id, display_name, result_only)
        for run_id in run_ids
    ]


import tkinter as tk


type_map = ["user_certain", "user_uncertain", "user_no_box"]


class Tracker1:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(
        self,
        name: str,
        parameter_name: str,
        dataset_name: str,
        run_id: int = None,
        display_name: str = None,
        result_only=False,
    ):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name
        self.cap = None  # 获取视频
        self.output_boxes = []  # 存储结果
        self.loop_flag = 0  # self.loop_flag 变量用于跟踪当前的视频帧位置。
        self.tracker = None
        self.pic = []
        self.control = []
        self.img_name = []
        self.display_name = ""
        self.frames = 0  # 总图片数
        env = env_settings()
        if self.run_id is None:
            self.results_dir = "{}/{}/{}".format(
                env.results_path, self.name, self.parameter_name
            )
        else:
            self.results_dir = "{}/{}/{}_{:03d}".format(
                env.results_path, self.name, self.parameter_name, self.run_id
            )
        if result_only:
            self.results_dir = "{}/{}".format(env.results_path, self.name)

        tracker_module_abspath = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "tracker", "%s.py" % self.name
            )
        )
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module(
                "lib.test.tracker.{}".format(self.name)
            )
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

    def create_tracker(self, params):
        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, "debug", 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output = {"target_bbox": [], "time": []}
        if tracker.params.save_all_boxes:
            output["all_boxes"] = []
            output["all_scores"] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {
            "target_bbox": init_info.get("init_bbox"),
            "time": time.time() - start_time,
        }
        if tracker.params.save_all_boxes:
            init_default["all_boxes"] = out["all_boxes"]
            init_default["all_scores"] = out["all_scores"]

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info["previous_output"] = prev_output

            if len(seq.ground_truth_rect) > 1:
                info["gt_bbox"] = seq.ground_truth_rect[frame_num]
            out = tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {"time": time.time() - start_time})

        for key in ["target_bbox", "all_boxes", "all_scores"]:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def draw_pic(self, frame_disp, pos):
        pos_l = "None"
        if self.control[pos] == 1:
            pos_l = "User"
        elif self.control[pos] == -1:
            pos_l = "Auto"
        label_type = self.output_boxes[pos][5]
        for font_width, font_color in ((3, (0, 0, 0)), (1, (255, 255, 255))):
            cv.putText(
                frame_disp,
                pos_l + " " + label_type,
                (20, 30),
                cv.FONT_HERSHEY_COMPLEX_SMALL,
                1.5,
                font_color,
                font_width,
            )
        if pos_l == "User":
            cv.rectangle(
                frame_disp,
                (0, 0),
                (frame_disp.shape[1], frame_disp.shape[0]),
                (0, 0, 255),
                20,
            )
        boxes = self.output_boxes[pos][1:]
        if boxes[4] != "user_no_box" or boxes[4] != "auto_no_box":
            for i in range(4):
                if isinstance(boxes[i], str):
                    boxes[i] = int(boxes[i])
            state = [boxes[0], boxes[1], boxes[2], boxes[3]]
            color = (0, 255, 0)
            if label_type[-9] == "u":
                color = (255, 0, 0)
            cv.rectangle(
                frame_disp,
                (state[0], state[1]),
                (state[2] + state[0], state[3] + state[1]),
                color,
                5,
            )
        return frame_disp

    def update_slider(self, display_name):  # 更新
        pos = cv.getTrackbarPos("start", display_name)
        pos = pos + 1
        cv.setTrackbarPos("start", display_name, pos)  # 将滑动条的位置设置为新的 值
        self.cap.set(cv.CAP_PROP_POS_FRAMES, pos)  # 让画面也移动到那一帧
        # print(cv.getTrackbarPos("start", display_name))

    def update_pic(self, display_name):  # 移动画面到那一帧
        frame_s = self.pic[cv.getTrackbarPos("start", display_name)]
        frame_e = []
        return frame_s, frame_e

    def lhdimshow(self, display_name, frame):
        pos = cv.getTrackbarPos("start", display_name)
        frame_disp = frame.copy()  # 防止修改原图
        frame_disp = self.draw_pic(frame_disp, pos)
        tipos = frame_disp.shape[0] - 15
        for font_width, font_color in ((3, (0, 0, 0)), (1, (255, 255, 255))):
            cv.putText(
                frame_disp,
                "Select r to init, then select blank space to make sure",
                (0, tipos),
                cv.FONT_HERSHEY_COMPLEX_SMALL,
                0.6,
                font_color,
                font_width,
            )
            cv.putText(
                frame_disp,
                "Select blank space to play, then blank space to stop. Press ESC quit.",
                (0, tipos + 10),
                cv.FONT_HERSHEY_COMPLEX_SMALL,
                0.6,
                font_color,
                font_width,
            )
        cv.imshow(display_name, frame_disp)

    def update_init(self, display_name, frame_disp):
        # 用户一开始设置追踪信号
        # 更新一次输出
        x, y, w, h = 0, 0, 0, 0
        frame_disp1 = frame_disp.copy()
        while w * h == 0:
            for font_width, font_color in ((3, (0, 0, 0)), (1, (255, 255, 255))):
                cv.putText(
                    frame_disp1,
                    "Draw Now!!!",
                    (20, 30),
                    cv.FONT_HERSHEY_COMPLEX_SMALL,
                    1.5,
                    font_color,
                    font_width,
                )
            self.lhdimshow(display_name, frame_disp1)
            x, y, w, h = cv.selectROI(
                display_name, frame_disp1, fromCenter=False
            )  # 添加额外的参数
            if (
                cv.getTrackbarPos(
                    "type:(0) user_certain, (1)user_uncertain, (2)user_no_box",
                    display_name,
                )
                == 2
            ):
                break

        label_type = type_map[
            cv.getTrackbarPos(
                "type:(0) user_certain, (1)user_uncertain, (2)user_no_box", display_name
            )
        ]

        pos = cv.getTrackbarPos("start", display_name)
        self.control[pos] = 1
        # print(label_type)
        if label_type != "user_no_box":
            init_state = [x, y, w, h]

            def _build_init_info(box):
                return {"init_bbox": box}

            self.tracker.initialize(frame_disp, _build_init_info(init_state))  # 初始化追踪器
            # 添加本次的输出
            init_state_res = init_state.copy()
            init_state_res.append(label_type)
            self.output_boxes[pos] = [self.output_boxes[pos][0]] + init_state_res
        else:
            self.output_boxes[cv.getTrackbarPos("start", display_name)][1] = 0
            self.output_boxes[cv.getTrackbarPos("start", display_name)][2] = 0
            self.output_boxes[cv.getTrackbarPos("start", display_name)][3] = 0
            self.output_boxes[cv.getTrackbarPos("start", display_name)][4] = 0
            self.output_boxes[cv.getTrackbarPos("start", display_name)][5] = label_type
        if label_type == "user_certain":
            label_type = "auto_certain"
        elif label_type == "user_uncertain":
            label_type = "auto_uncertain"
        elif label_type == "user_no_box":
            label_type = "auto_no_box"

        return label_type

    def auto_draw(self, display_name, label_type, frame):
        out = self.tracker.track(frame)  # self.display_name = display_name返回画的图
        state = [int(s) for s in out["target_bbox"]]
        state.append(label_type)
        pos = cv.getTrackbarPos("start", display_name)
        self.output_boxes[pos] = [self.output_boxes[pos][0]] + state
        return frame

    def run_lhd(
        self,
        videofilepath,
        optional_box=None,
        debug=None,
        visdom_info=None,
        save_results=False,
        img_name=[],
    ):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """
        self.img_name = img_name
        params = self.get_parameters()
        debug_ = debug
        if debug is None:
            debug_ = getattr(params, "debug", 0)
        params.debug = debug_
        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)
        multiobj_mode = getattr(
            params,
            "multiobj_mode",
            getattr(self.tracker_class, "multiobj_mode", "default"),
        )
        if multiobj_mode == "default":
            self.tracker = self.create_tracker(params)
        elif multiobj_mode == "parallel":
            self.tracker = MultiObjectWrapper(
                self.tracker_class, params, self.visdom, fast_load=True
            )
        else:
            raise ValueError("Unknown multi object mode {}".format(multiobj_mode))
        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"
        self.cap = cv.VideoCapture(videofilepath)  # 打开视频文件
        fps = self.cap.get(cv.CAP_PROP_FPS)
        display_name = "Display: " + self.tracker.params.tracker_name
        self.display_name = display_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)  # 设置窗口
        cv.resizeWindow(display_name, 960, 720)  # 窗口大小
        success, frame = self.cap.read()

        def nothing(emp):
            pass

        self.frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        print("一共有：", self.frames)
        cv.createTrackbar("start", display_name, 0, self.frames - 1, nothing)  # 创建进度条
        cv.createTrackbar(
            "type:(0) user_certain, (1)user_uncertain, (2)user_no_box",
            display_name,
            0,
            2,
            nothing,
        )  # 创建进度条
        if self.img_name == []:  # 输入到是视频的话
            self.img_name = [x for x in range(self.frames)]
        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            cv.imshow(display_name, frame)  # 读取第一帧
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            self.tracker.initialize(frame, {"init_bbox": optional_box})
            self.output_boxes.append(optional_box)

        key = None
        while success:
            self.pic.append(frame)
            success, frame = self.cap.read()

        self.results_dir = os.path.join(
            os.path.dirname(os.path.abspath(videofilepath)), "label"
        )
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        video_name = Path(videofilepath).stem
        base_results_path = os.path.join(self.results_dir, video_name)
        bbox_file = "{}.txt".format(base_results_path)

        if os.path.exists(bbox_file):  # 如果已经读到了
            with open(bbox_file, "r") as file:
                lines = file.readlines()
                for line in lines:
                    line = line
                    line = line.strip().split()  # 去除换行符并按空格拆分
                    self.output_boxes.append(line)
            control = [1 if "user" in x[5] else -1 for x in self.output_boxes]
        else:  # 如果没有读到了
            control = [0 for x in range(self.frames)]
            self.output_boxes = [
                [x] + y
                for x, y in zip(
                    self.img_name, [[0, 0, 0, 0, "auto_no_box"]] * self.frames
                )
            ]  # 初始化输出
        pos_s, pos_e = -1, self.frames

        self.control = control

        while key != 27:  # 播放图片 按esc退出
            key = cv.waitKey(1)
            frame_s, frame_e = self.update_pic(display_name)
            if cv.getTrackbarPos("start", display_name) != pos_s:
                pos_s = cv.getTrackbarPos("start", display_name)
                self.lhdimshow(display_name, frame_s)
            if key == ord("r"):
                label_type = self.update_init(display_name, frame_s)  # 手动画图,获取当前的标签状态
                next_pos_l = self.control[cv.getTrackbarPos("start", display_name) + 1]
                while (
                    key != 32
                    and cv.getTrackbarPos("start", display_name) < self.frames - 1
                    and next_pos_l != 1
                ):
                    self.update_slider(display_name)  # 更新进度条的位置与实际播放位置，自动增加进度
                    frame = self.pic[cv.getTrackbarPos("start", display_name)]
                    self.control[cv.getTrackbarPos("start", self.display_name)] = -1
                    if label_type != "auto_no_box":
                        frame = self.auto_draw(
                            display_name, label_type, frame
                        )  # 自动分割一张画图
                    else:
                        self.output_boxes[cv.getTrackbarPos("start", display_name)][
                            1
                        ] = 0
                        self.output_boxes[cv.getTrackbarPos("start", display_name)][
                            2
                        ] = 0
                        self.output_boxes[cv.getTrackbarPos("start", display_name)][
                            3
                        ] = 0
                        self.output_boxes[cv.getTrackbarPos("start", display_name)][
                            4
                        ] = 0
                        self.output_boxes[cv.getTrackbarPos("start", display_name)][
                            5
                        ] = label_type
                    self.lhdimshow(display_name, frame)
                    if cv.getTrackbarPos("start", display_name) + 1 < self.frames:
                        next_pos_l = self.control[
                            cv.getTrackbarPos("start", display_name) + 1
                        ]
                    key = cv.waitKey(1)
                print("标记结束")
            elif key == 32:  # 暂停检查
                key = None
                print("播放")
                while (
                    key != 32
                    and cv.getTrackbarPos("start", display_name) < self.frames - 1
                ):  # 不断地显示画面
                    frame = self.pic[cv.getTrackbarPos("start", display_name)]
                    self.lhdimshow(display_name, frame)
                    self.update_slider(display_name)  # 更新进度条的位置与实际播放位置，自动增加进度
                    key = cv.waitKey(1)
                    time.sleep(1 / fps)
        # When everything done, release the capture
        self.cap.release()
        cv.destroyAllWindows()

        if save_results:
            print(base_results_path)

            with open(bbox_file, "w") as file:
                # 将列表中的每个元素转换为字符串并写入文件
                for item in self.output_boxes:
                    file.write(" ".join(map(str, item)))
                    file.write("\n")

    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module(
            "lib.test.parameter.{}".format(self.name)
        )
        params = param_module.parameters(self.parameter_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")
