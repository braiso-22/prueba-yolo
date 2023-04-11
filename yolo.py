from image_utils import Camera
import cv2 as cv
from ultralytics import YOLO


def get_cpu_numpy(box_item):
    return box_item.cpu().numpy()


def plot_bboxes(frame, params):
    model = params['model']
    xyxys = []
    confidences = []
    class_ids = []
    results = model.predict(
        source=frame,
        conf=0.25,
    )
    result = results[0]
    if len(result.boxes) == 0:
        return frame, None
    boxes = result.boxes[0]
    xyxy = get_cpu_numpy(boxes.xyxy)[0]
    confidence = get_cpu_numpy(boxes.conf)[0]
    if len(boxes.cls) != 0:
        class_id = result.names[int(boxes.cls[0])]

    x1, y1, x2, y2 = xyxy
    cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv.rectangle(frame, (int(x1), int(y1)), (int(x1) + 100, int(y1) + 20), (0, 255, 0), -1)
    cv.putText(frame, class_id, (int(x1), int(y1) + 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    print("boxes: ", xyxy)
    print("confidences: ", confidence)
    print("class_ids: ", class_id)

    return frame, None


def main():
    # load model
    model = YOLO('best.pt')
    # load video
    option = input("Press 1 to start video capture\nPress 2 to start video capture from file\n")
    if option == "1":
        Camera.video_capture(plot_bboxes, {"model": model})
    elif option == "2":
        Camera.video_capture(
            plot_bboxes,
            {"model": model},
            video_path="videos/video3.mp4"
        )
    else:
        print("Invalid option")


if __name__ == "__main__":
    main()
