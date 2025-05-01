
# 🧠 ASL Gesture Recognition using OpenCV & SVM 🤟

## 📄 Description
A real-time American Sign Language (ASL) gesture recognition system using **Python**, **OpenCV**, and **Support Vector Machines (SVM)**.  
This project uses **HOG features** for image processing and an SVM classifier to recognize hand gestures via webcam input.

## 🚀 Features
- 📷 Live webcam-based gesture recognition  
- ✂️ ROI (Region of Interest) extraction  
- 🧩 HOG (Histogram of Oriented Gradients) feature extraction  
- 🧠 SVM-based model training & prediction  
- 💾 Saves trained model for reuse  

## 📦 Dataset
To run this project, you need the ASL Alphabet dataset:

🔗 [Download ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (from Kaggle)

> Extract the dataset and place the folders as:
```
/asl_alphabet_train
/asl_alphabet_test
```

## 🛠️ Requirements
- Python 3.x
- OpenCV (`cv2`)
- NumPy

Install dependencies using:
```bash
pip install opencv-python numpy
```

## ▶️ How to Run
1. Download and extract the dataset as described above.
2. Run the script:
```bash
python main.py
```
3. Show a hand gesture inside the green ROI box on the webcam.
4. Press `q` to exit.

## 📂 Output
- Trained model saved as `asl_gesture_model.xml`
- Real-time window showing prediction and highlighted ROI

## 🙌 Acknowledgements
- Dataset provided by Kaggle user **grassknoted**
