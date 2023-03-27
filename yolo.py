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
        imgsz=640,
        conf=0.25,
    )
    for result in results:
        boxes = result.boxes
        xyxys.append(get_cpu_numpy(boxes.xyxy))
        confidences.append(get_cpu_numpy(boxes.conf))
        # class_ids.append(get_cpu_numpy(result.names))
    # print in frame a rectangle with the boxes
    for xys in xyxys[0]:
        cv.rectangle(frame, (int(xys[0]), int(xys[1])), (int(xys[2]), int(xys[3])), (0, 255, 0), 2)

    print("boxes: ", xyxys)
    print("confidences: ", confidences)
    print("class_ids: ", class_ids)

    return frame, None


def main():
    # load model
    model = YOLO('yolov8n.pt')
    # load video
    Camera.video_capture(plot_bboxes, {"model": model})


if __name__ == "__main__":
    main()
