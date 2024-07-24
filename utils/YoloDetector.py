from ultralytics import YOLO
import cv2
from PIL import Image
import matplotlib.pyplot as plt


class YoloDetector(YOLO):
    def __init__(self, model: str = "yolov8n.pt", task: str = None, verbose: bool = False):
        super().__init__(model=model, task=task, verbose=verbose)
        self.classes = None
        self.result_image = None

    def _get_colours(self, cls_num):
        base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        color_index = cls_num % len(base_colors)
        increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
        color = [base_colors[color_index][i] + increments[color_index][i] *
                 (cls_num // len(base_colors)) % 256 for i in range(3)]
        return tuple(color)

    def detect(self, image_path: str):
        results = super().predict(image_path)
        for result in results:
            # get the classes names
            classes_names = result.names
            frame = cv2.imread(image_path)
            original = frame.copy()
            for box in result.boxes:
                # check if confidence is greater than 40 percent
                if box.conf[0] > 0.4:
                    # get coordinates
                    [x1, y1, x2, y2] = box.xyxy[0]
                    # convert to int
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # get the class
                    cls = int(box.cls[0])

                    # get the class name
                    class_name = classes_names[cls]

                    # get the respective colour
                    colour = self._get_colours(cls)

                    # draw the rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                    # put the class name and confidence on the image
                    text = f'{class_name} {box.conf[0]:.2f}'
                    text_size = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    text_x = x1
                    text_y = y1 - text_size[1] - 5

                    # draw the rectangle behind the text
                    cv2.rectangle(frame, (text_x - 5, text_y - 5), (text_x +
                                  text_size[0] + 5, text_y + text_size[1] + 5), colour, -1)

                    # put the class name and confidence on the image
                    cv2.putText(frame, text, (text_x, text_y + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            frame = cv2.vconcat(
                [original, frame]) if frame.shape[0] < frame.shape[1] else cv2.hconcat([original, frame])
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_frame)
            self.result_image = pil_image

    def show(self):
        plt.imshow(self.result_image)
        plt.axis('off')  # Hide the axis
        plt.show()
