# Vehicle Detection Using HOG + SVM (Histogram of Oriented Gradients)

This project demonstrates a **classical machine learning pipeline** for detecting vehicles using **Histogram of Oriented Gradients (HOG)** for feature extraction and a **Support Vector Machine (SVM)** for classification.

The goal is to build a **lightweight, interpretable, and efficient vehicle detection system** that mirrors early perception modules used in **autonomous vehicles (AVs)** and **advanced driver assistance systems (ADAS)** before deep learning became the dominant approach.

This project offers hands-on insight into how object detection was traditionally performed using engineered features, providing the foundation for understanding modern perception systems.

---

## Business Statement

To develop a simple, efficient, and interpretable **vehicle detection system** using HOG for feature extraction and SVM for classification.  

This classical approach provides:

- A **baseline perception model** for autonomous driving research.
- A lightweight alternative to deep learning models in computationally constrained environments.
- A practical demonstration of object detection without neural networks.

---

### 2. HOG Feature Extraction

HOG captures gradient orientation patterns, allowing the model to recognize:

- Edges  
- Shapes  
- Vehicle structures  

The process includes:

- Defining orientations  
- Selecting cell sizes  
- Block normalization  
- Visualization of HOG features for interpretability  

The extracted HOG vectors represent each image in a format suitable for SVM training.

---

### 3. Support Vector Machine (SVM) Classification

A Support Vector Classifier is trained to distinguish between:

- **1 → Car**  
- **0 → Non-car**

Training steps include:

- Model initialization  
- GridSearchCV hyperparameter optimization  
- Performance evaluation  
- Cross-validation to measure generalization  

---

### 4. Model Optimization (GridSearchCV)

Hyperparameters tuned include:

- `C`
- `gamma`
- `kernel`

After optimization, the model reached **99% accuracy** and **98.89% cross-validation accuracy**, showing excellent generalization and reliability.

---

### 5. Evaluation Metrics

The system outputs:

- Precision  
- Recall  
- F1-score  
- Confusion matrix  
- Cross-validation accuracy  

These metrics ensure that the model is both **accurate and consistent** critical in AV perception where mistakes can be costly.

---

## Observations

- HOG features were highly effective at capturing vehicle structure.
- Hyperparameter tuning improved accuracy from **96% → 99%**.
- The model showed excellent stability across cross-validation folds.
- Despite its simplicity, HOG/SVM performs very well in controlled environments.

However:

- Performance may degrade in complex real-world scenes (shadows, curves, occlusions).

---

## Future Improvements

Possible directions for extending this project:

### 1. Sliding Window Object Detection
Move from classification to **vehicle localization** by scanning full images with sliding windows.

### 2. Hard Negative Mining
Improve training by adding difficult non-vehicle examples.

### 3. Deep Learning Comparison
Evaluate performance against:
- YOLO  
- Faster R-CNN  
- SSD  
- Transformer-based detectors  


### 4. Simulation Testing
Test performance on:
- CARLA simulator  
- KITTI dataset  
- Driving videos
- Airsim
- Autoware

---

## Requirements

- `numpy`  
- `opencv-python`  
- `matplotlib`   
- `scikit-learn`  
- `seaborn`  

---

## About Me

I am a data science and machine learning practitioner with a strong passion for **autonomous vehicles** and **computer vision**, focused on building solutions that bridge research and real-world deployment.

Feel free to connect with me:

- **LinkedIn:** https://www.linkedin.com/in/patrickedosoma  
- **Email:** edosomapatrick41@gmail.com  

---

## Contributions

Contributions and suggestions are welcome!  
Please open an issue or submit a pull request.

---

## ⭐ Star This Repo

If you found this project helpful, please star ⭐ the repository to show support!

