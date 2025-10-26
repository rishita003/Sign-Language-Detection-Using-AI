# Hand Sign Detection Using AI

A deep learning-based hand sign recognition system using Python, TensorFlow/Keras, and computer vision. This project can recognize hand signs from images or real-time webcam input.

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Creation](#dataset-creation)
- [Model Training](#model-training)
- [Running Inference](#running-inference)
- [Project Structure](#project-structure)
- [License](#license)

---

## Features

- Recognizes hand gestures/signs from images or webcam input.
- Custom dataset creation using your own hand gestures.
- Trainable convolutional neural network (CNN) for classification.
- Real-time gesture detection and classification.

---

## Prerequisites

- Python 3.8+  
- TensorFlow 2.x / Keras  
- OpenCV  
- NumPy, Matplotlib  
- Other dependencies listed in `requirements.txt`

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/rishita003/Sign-Language-Detection-Using-AI.git
cd Sign-Language-Detection-Using-AI
````

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## Dataset Creation

You can create your own hand gesture dataset using the provided scripts.

1. **Collect images**:

```bash
python collect_imgs.py
```

* Follow the instructions to capture images for each gesture.
* Images will be saved in the `Dataset/` folder structured by class.

2. **Check your dataset**:

* Ensure that each gesture/class has enough images for training.

---

## Model Training

Train the CNN model using the collected dataset:

```bash
python train_classifier.py
```

* The model will be trained and saved as `cnn_model.h5`.
* Hyperparameters can be adjusted in the script (epochs, batch size, learning rate).

---

## Running Inference

Once the model is trained, you can test it on new images or webcam input:

```bash
python inference_classifier.py
```

* For real-time webcam detection, follow the on-screen instructions.
* The system will classify your hand gestures and display results live.

---

## Project Structure

```
Sign-Language-Detection-Using-AI/
│
├── collect_imgs.py         # Script to collect gesture images
├── create_dataset.py       # Organize images into dataset format
├── train_classifier.py     # Train CNN model
├── inference_classifier.py # Run trained model on webcam/input
├── cnn_model.h5            # Saved trained model
├── requirements.txt        # Python dependencies
├── Dataset/                # Folder to store gesture images
└── README.md               # Project documentation
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Notes / Tips

* Make sure lighting is consistent during image collection for better accuracy.
* Use `requirements.txt` to install all dependencies.
* Increase dataset size for better model performance.
* You can modify `train_classifier.py` to tweak CNN architecture or hyperparameters.

```
Do you want me to do that?
```
