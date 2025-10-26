# Hand Sign Detection using AI

This project is a real-time **Hand Sign Detection system** using **Mediapipe** and **Random Forest classifier**. It recognizes hand gestures (A, B, L by default) from webcam input and predicts the corresponding class.

---

## **Table of Contents**

- [Features](#features)  
- [Requirements](#requirements)  
- [Setup](#setup)  
- [Usage](#usage)  
  - [1. Clone the Repository](#1-clone-the-repository)  
  - [2. Install Dependencies](#2-install-dependencies)  
  - [3. Collect Data (Optional)](#3-collect-data-optional)  
  - [4. Create Dataset](#4-create-dataset)  
  - [5. Train the Model](#5-train-the-model)  
  - [6. Run Inference](#6-run-inference)  
- [Dataset](#dataset)  
- [Notes](#notes)  
- [License](#license)  

---

## **Features**

- Real-time hand sign detection using webcam.  
- Uses **Mediapipe** for hand landmark extraction.  
- Trains a **Random Forest classifier** on hand landmark features.  
- Modular scripts for data collection, feature extraction, training, and inference.  

---

## **Requirements**

- Python 3.10+  
- Packages:
  ```bash
  pip install opencv-python mediapipe matplotlib scikit-learn numpy


* Webcam (built-in or external)

---

## **Setup**

### **1. Clone the Repository**

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

---

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is missing, manually install:

```bash
pip install opencv-python mediapipe matplotlib scikit-learn numpy
```

---

### **3. Collect Data (Optional)**

If you want to collect your own hand gestures:

```bash
python collect_images.py
```

* Press **`q`** to start collecting frames for each class.
* Images are stored in `./data/<class_number>/`.
* By default, the script collects **100 images per class**.
* Update `number_of_classes` and `dataset_size` in the script as needed.

> **Tip:** You can also use a pre-existing dataset (like ASL) instead of collecting images.

---

### **4. Create Dataset**

Extract hand landmark features from collected images:

```bash
python create_dataset.py
```

* Generates `data.pickle` containing:

  * `data` → hand landmark features
  * `labels` → class labels

---

### **5. Train the Model**

Train the classifier:

```bash
python train_classifier.py
```

* Splits the dataset into **80% train / 20% test**.
* Prints accuracy on the test set.
* Saves the trained model as `model.p`.

---

### **6. Run Inference**

Run real-time hand sign detection using your webcam:

```bash
python inference_classifier.py
```

* Make sure `model.p` exists in the project folder.
* Predictions appear as bounding boxes and labels on the webcam feed.
* Press **`ESC`** or close the window to stop.

---

## **Dataset**

* By default, the project works with 3 classes (0, 1, 2 → mapped to `A`, `B`, `L`).
* You can use your own data or any public dataset like **ASL** (A-Z).
* If using a larger dataset, update:

  * `number_of_classes` in `collect_images.py`
  * `labels_dict` in `inference_classifier.py`

---

## **Notes**

* **Webcam index:** Default is `0` in scripts. Change it if your webcam uses a different index:

  ```python
  cap = cv2.VideoCapture(0)  # change 0, 1, 2, etc.
  ```
* **Model improvements:** You can replace Random Forest with other classifiers or deep learning models for better accuracy.
* **Data augmentation:** Recommended if using a small dataset.
 
