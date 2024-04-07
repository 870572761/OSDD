import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), "..")
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import Tracker1


def run_video(
    tracker_name,
    tracker_param,
    videofile,
    img_name,
    optional_box=None,
    debug=None,
    save_results=False,
):
    """Run the tracker on your webcam.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
    """
    tracker = Tracker1(tracker_name, tracker_param, "video")
    pic,item = tracker.run_lhd(
        videofilepath=videofile,
        optional_box=optional_box,
        debug=debug,
        save_results=save_results,
        img_name=img_name,
    )
    return pic,item


import cv2


def images_to_video(image_folder, video_name, fps):
    images_name = []
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images = sorted(images)
    images_name = [img for img in images if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    return images_name


def main():
    parser = argparse.ArgumentParser(description="Run the tracker on your webcam.")
    parser.add_argument(
        "--tracker_name", type=str, default="ostrack", help="Name of tracking method."
    )
    parser.add_argument(
        "--tracker_param",
        type=str,
        default="vitb_256_mae_32x4_ep300",
        help="Name of parameter file.",
    )
    parser.add_argument(
        "--class_name", type=str, default="jizhan", help="Name of class."
    )
    parser.add_argument("--video", type=bool, default=False)
    parser.add_argument(
        "--imagedir",
        type=str,
        default="/home/lei/pj2/OSTrack-main/data/cam13",
        help="path to a video file or filedirs.",
    )
    parser.add_argument(
        "--videofile",
        type=str,
        default="data/test.mp4",
        help="path to a video file or filedirs.",
    )
    parser.add_argument(
        "--optional_box",
        type=float,
        default=None,
        nargs="+",
        help="optional_box with format x y w h.",
    )
    parser.add_argument("--debug", type=int, default=0, help="Debug level.")
    parser.add_argument(
        "--save_results",
        dest="save_results",
        action="store_true",
        help="Save bounding boxes",
    )
    parser.set_defaults(save_results=True)

    img_name = []

    args = parser.parse_args()
    if args.video != True:
        args.videofile = os.path.join(
            args.imagedir,
            args.class_name + ".mp4",
        )
        img_name = images_to_video(args.imagedir, args.videofile, 25)
    pic,items = run_video(
        args.tracker_name,
        args.tracker_param,
        args.videofile,
        img_name,
        args.optional_box,
        args.debug,
        args.save_results,
    )
    pic_save_path = os.path.dirname(args.videofile)
    if args.video:#保存图片
        for (name,img) in zip(items,pic):
            name = name[0]
            if name %100==0:
                print("Save",name)
            img_path = os.path.join(pic_save_path,str(name)+'.jpg')
            cv2.imwrite(img_path, img)


if __name__ == "__main__":
    main()
