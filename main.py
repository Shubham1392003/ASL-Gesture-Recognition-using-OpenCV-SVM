import cv2
import numpy as np
import os

# Configuration
DATASET_PATH = 'asl_alphabet_train'
TEST_PATH = 'asl_alphabet_test'
IMG_SIZE = (64, 64)  # Must be tuple for OpenCV
ROI_SIZE = 200

# Corrected HOG initialization
hog = cv2.HOGDescriptor(
    _winSize=IMG_SIZE,
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9
)

def load_dataset(path):
    features = []
    labels = []
    class_names = sorted(os.listdir(path))
    
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(path, class_name)
        print(f"Loading {class_name}...")
        
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, IMG_SIZE)
            
            hog_features = hog.compute(img)
            features.append(hog_features.flatten())
            labels.append(label_idx)
    
    return np.array(features, dtype=np.float32), np.array(labels), class_names

# Rest of the code remains the same...

# Load training data
print("Loading training data...")
train_features, train_labels, class_names = load_dataset(DATASET_PATH)

# Train SVM classifier
print("Training classifier...")
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.train(train_features, cv2.ml.ROW_SAMPLE, train_labels)

# Save the trained model
svm.save('asl_gesture_model.xml')

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get region of interest (center of frame)
    height, width = frame.shape[:2]
    x = width//2 - ROI_SIZE//2
    y = height//2 - ROI_SIZE//2
    roi = frame[y:y+ROI_SIZE, x:x+ROI_SIZE]

    # Preprocess ROI
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_resized = cv2.resize(roi_gray, IMG_SIZE)
    
    # Extract HOG features
    hog_features = hog.compute(roi_resized).flatten().astype(np.float32)
    
    # Predict gesture
    _, result = svm.predict(hog_features.reshape(1, -1))
    prediction = class_names[int(result[0, 0])]

    # Display results
    cv2.rectangle(frame, (x, y), (x+ROI_SIZE, y+ROI_SIZE), (0, 255, 0), 2)
    cv2.putText(frame, f"Prediction: {prediction}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('ASL Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()