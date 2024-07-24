# YOLO Object Detection with Explainable AI Methods

## Description

This project provides a Python implementation for object detection using YOLO (You Only Look Once) models. It also includes several methods for Explainable AI (XAI) to provide insights into the decisions made by the model. The project integrates YOLO for detecting objects in images and includes various Explainable AI techniques to understand the model’s predictions.

### Features

-   **YOLO Object Detection**: Uses YOLO models for real-time object detection.
-   **Explainable AI Methods**: Implements Layer-wise Relevance Propagation (LRP) and several CAM-based models to explain and visualize model predictions.
-   **Trained Model**: Includes a pre-trained YOLO model for immediate use.
-   **Dataset**: The dataset used for training can be downloaded through the provided Jupyter Notebook.
-   **Main Code**: Contains the main code and explanations in the Jupyter Notebook.

## Requirements

To set up this project, make sure the required libraries are installed by using `pip`:

`pip install -r requirements.txt`

## Usage

1. **Run Object Detection**

    Use the provided Jupyter Notebook to load the pre-trained YOLO model and run object detection on images. The notebook contains instructions for running inference and visualizing results.

2. **Understand Explainable AI Methods**

    The project includes implementations of several Explainable AI techniques to interpret the model's predictions:

    - **Layer-wise Relevance Propagation (LRP)**: LRP is a technique used to decompose the model's prediction into relevance scores assigned to each input feature. It helps in understanding which parts of the input image are most influential in making the prediction.

    - **High-Resolution Class Activation Mapping (HiResCAM)**: HiResCAM is an enhancement of CAM that produces high-resolution visual explanations by element-wise multiplying the activations with the gradients. It provides more detailed visualizations of the model's focus areas.

    - **Layer CAM**: Layer CAM is a spatially weighted class activation mapping technique that improves the explanation quality, especially in lower layers of the network.

    - **XGradCAM**: XGradCAM extends GradCAM by scaling the gradients by normalized activations, which helps in producing more stable and interpretable visualizations.

    - **EigenGradCAM**: EigenGradCAM combines EigenCAM and GradCAM, leveraging the principal components of activations and gradients for class-specific explanations.

    - **EigenCAM**: EigenCAM uses the first principal component of 2D activations to generate explanations. Although it doesn’t involve class discrimination, it often yields useful results.

    - **GradCAM**: GradCAM is a popular technique that weights the activations by the average gradients of the output with respect to the feature maps. It provides visualizations highlighting the regions most responsible for the model's decision.

## References

-   [YOLO: You Only Look Once](https://arxiv.org/abs/1506.02640)
-   [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
-   [Easy Explain](https://github.com/stavrostheocharis/easy_explain)
-   [Easy Explain: Explainable AI for YOLOv8](https://towardsai.net/p/machine-learning/easy-explain-explainable-ai-for-yolov8)
-   [Interpreting Deep Learning Models for Computer Vision](https://medium.com/google-developer-experts/interpreting-deep-learning-models-for-computer-vision-f95683e23c1d)
-   [Drug discovery with explainable artificial intelligence](https://www.nature.com/articles/s42256-020-00236-4)
