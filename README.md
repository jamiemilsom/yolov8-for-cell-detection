# YOLOv8-Based Cell Detection 🦠

This project focuses on the development of a cell detection model using YOLOv8, a state-of-the-art object detection architecture. The model is designed to accurately identify and localize cells within microscopy images.

## Key Features:

*   **CVAT for Annotation:**  The project utilizes CVAT (Computer Vision Annotation Tool), a powerful open-source platform, for generating high-quality annotations of cell locations within the images.
*   **YOLOv8 Framework:** Leverages the speed and accuracy of the YOLOv8 model for real-time or high-throughput cell detection.
*   **Customizable:** The model can be easily adapted to different cell types and microscopy settings.
*   **Python Implementation:**  Training, validation, and inference are implemented in Python using the Ultralytics YOLOv8 library.

## Dataset:

*   **Image Source:** The dataset comprises microscopy images of cells.
*   **Annotation Format:** Images are annotated in CVAT, then exported to YOLO format for model training.
*   **Dataset Structure:**
    *   `cell_data/images/train/`: Contains training images.
    *   `cell_data/images/test/`: Contains test images (used for evaluation only).
    *   `cell_data/labels/train/`: Contains YOLO format annotations for training images.
    *   `config.yaml`: Defines dataset paths, class names, and augmentation settings.

## Usage:

1.  **Installation:** 
    *   Ensure you have Python installed.
    *   Install the required libraries:
        ```bash
        pip install ultralytics opencv-python
        ```
2.  **Training:** Run `model_training.py` to train the YOLOv8 model using the `config.yaml` file and the provided training data. The trained model will be saved in the `runs/` directory.
3.  **Inference:**  Run `model_testing.py` to load the trained model and perform inference on images in the `cell_data/images/test` directory.
    *   The script will display the images with predicted bounding boxes and confidence scores.
    *   Press 'q' to move to the next image.
    *   Feel free to add your own images to images/test to evaluate model performance.


## Customisation:

*   Adjust the `config.yaml` file to modify dataset paths, class names, and augmentation parameters.
*   Explore different YOLOv8 model architectures (e.g., yolov8s, yolov8m) for varying levels of speed and accuracy.

## Future Work:

*   Incorporate additional cell types or features for detection.
*   Experiment with different model architectures and hyperparameters to improve performance.
*   Integrate the model into a larger analysis pipeline for automated cell counting and tracking.

## Contributing:

Contributions are welcome! Feel free to open issues, submit pull requests, or provide feedback to help improve this project.
