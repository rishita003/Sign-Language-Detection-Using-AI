
# Hand Gesture Recognition Using CNN & OpenCV

This project implements a real-time **hand gesture recognition system** using OpenCV and Convolutional Neural Networks (CNN) in Keras/TensorFlow. The system captures hand gestures via webcam, trains a CNN model, and predicts gestures in real-time.

---

## Features

- Real-time gesture recognition using webcam
- CNN-based classification of hand gestures
- Custom dataset creation and augmentation
- Supports text input and calculator modes via gestures
- Voice feedback using text-to-speech (optional)

---

## Folder Structure


HandGestureRecognition/
│
├── gestures/                  # Stores captured gesture images per class
│   ├── 0/
│   ├── 1/
│   └── ...
├── cnn_model_train.py          # Trains the CNN model
├── create_gestures.py          # Captures hand gestures from webcam
├── rotate_images.py            # Data augmentation (flip images)
├── load_images.py              # Prepares pickle datasets for training
├── set_hand_histogram.py       # Captures hand histogram for segmentation
├── final.py                    # Real-time gesture recognition & prediction
├── hist                        # Generated hand histogram
├── cnn_model_keras2.h5         # Trained CNN model
├── train_images, train_labels
├── val_images, val_labels
└── test_images, test_labels

````

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- TensorFlow 2.x
- Keras
- scikit-learn
- pyttsx3 (for voice feedback)

Install dependencies using:

```bash
pip install opencv-python numpy tensorflow keras scikit-learn pyttsx3
````

---

## Usage

Follow the steps below in order:

### 1. Set Hand Histogram

Calibrate your hand color for proper segmentation.

```bash
python set_hand_histogram.py
```

* Place your hand inside the green squares.
* Press **`c`** to capture histogram.
* Press **`s`** to save and exit.
* This creates a `hist` file used in gesture capture and recognition.

---

### 2. Capture Gestures

Collect images for each gesture.

```bash
python create_gestures.py
```

* Input **gesture number** and **gesture name**.
* Capture 1200 images per gesture by following on-screen instructions.

---

### 3. Augment Data (Optional)

Flip images horizontally to increase dataset size:

```bash
python rotate_images.py
```

---

### 4. Prepare Datasets

Convert captured images to pickle files for training:

```bash
python load_images.py
```

This generates:

* `train_images`, `train_labels`
* `val_images`, `val_labels`
* `test_images`, `test_labels`

---

### 5. Train CNN Model

Train the model and save it:

```bash
python cnn_model_train.py
```

* Trains a CNN for 15 epochs.
* Saves the trained model as `cnn_model_keras2.h5`.

---

### 6. Run Real-Time Gesture Recognition

Use the trained model to detect gestures in real-time:

```bash
python final.py
```

* Supports **text mode** and **calculator mode**.
* Voice feedback is enabled by default. Toggle voice with **`v`**.
* Exit modes using **`q`** or **`c`/`t`** as indicated on-screen.

---

## Notes

* Ensure proper lighting for accurate hand segmentation.
* Webcam index may vary. Default is `0`. Change in `cv2.VideoCapture(0)` if needed.
* All gestures must be collected before training the model.
* Model is trained on grayscale 50x50 images.

---

## License

This project is for **educational and research purposes**. You may modify and distribute the code but credit the original author.

---

## Acknowledgments

* OpenCV for computer vision
* Keras and TensorFlow for deep learning
* pyttsx3 for text-to-speech feedback



