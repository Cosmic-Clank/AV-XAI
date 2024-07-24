from typing import Literal
from easy_explain import YOLOv8LRP
from YOLOv8_Explainer import yolov8_heatmap, display_images
from PIL import Image
import torchvision
from matplotlib import pyplot as plt
from ultralytics import YOLO
import torch


class YoloExplainer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.class_names = list(
            class_name for class_name in self.model.names.values())

    def explain(self, image_path: str, explaination_method: Literal["LRP", 'GradCAM', 'GradCAM++', 'XGradCAM', 'EigenCAM', 'HiResCAM', 'LayerCAM', 'EigenGradCAM']):
        if explaination_method == 'LRP':
            image = Image.open(image_path)
            desired_size = (512, 640)
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(desired_size),
                torchvision.transforms.ToTensor(),
            ])
            image = transform(image)

            lrp = YOLOv8LRP(self.model, power=2, eps=1, device='gpu')
            explanation_lrp = lrp.explain(
                image, cls=self.class_names[0], contrastive=False).cpu()
            plt.imshow(explanation_lrp)
            plt.xticks([])
            plt.yticks([])

            plt.tight_layout()
            plt.show()
        elif explaination_method in ['GradCAM', 'GradCAM++', 'XGradCAM', 'EigenCAM', 'HiResCAM', 'LayerCAM', 'EigenGradCAM']:
            model = yolov8_heatmap(
                weight=self.model_path,
                conf_threshold=0.4,
                device=torch.device(
                    "cuda:0" if torch.cuda.is_available() else "cpu"),
                method=explaination_method,
                layer=[10, 12, 14, 16, 18, -3],
                # backward_type="all",
                ratio=0.02,
                show_box=False,
                renormalize=False,
            )

            imagelist = model(
                img_path=image_path,
            )

            display_images(imagelist)
        else:
            raise ValueError(
                "Invalid explaination method. Choose from: ['LRP', 'GradCAM', 'GradCAM++', 'XGradCAM', 'EigenCAM', 'HiResCAM', 'LayerCAM', 'EigenGradCAM']")
