import numpy as np
import os
import cv2
import sqlite3
import hashlib
import time
from numpy.linalg import norm
from numpy import dot
from sklearn.model_selection import train_test_split

DB_NAME = 'database.db'

def create_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL,
            image_hash TEXT NOT NULL UNIQUE
        )
    ''')
    conn.commit()
    conn.close()

def insert_face(label, image_hash):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute('INSERT INTO faces (label, image_hash) VALUES (?, ?)', (label, image_hash))
        conn.commit()
    except sqlite3.IntegrityError:
        pass  # Duplicate image, ignore
    conn.close()

def load_face_images_from_folder(folder_path, image_size=(100, 100)):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.jpg'):
            label = os.path.splitext(filename)[0].strip()
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)
            if len(faces) == 0:
                continue
            (x, y, w, h) = faces[0]
            face_img = img[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, image_size)
            images.append(face_resized)
            labels.append(label)
            print(f"{label}: {face_resized.shape}")
    return images, labels

def hash_image(img):
    img_bytes = img.tobytes()
    return hashlib.sha256(img_bytes).hexdigest()

def train_eigenfaces(images, max_components=10):
    data = np.array([img.flatten() for img in images], dtype=np.float32)
    mean, eigenvectors = cv2.PCACompute(data, mean=None, maxComponents=max_components)
    print(f"Data shape for PCA: {data.shape}")
    return mean, eigenvectors

def project_face(face, mean, eigenvectors):
    face_vector = face.flatten().astype(np.float32).reshape(1, -1)  
    mean = mean.reshape(1, -1)                                      
    mean_subtracted = face_vector - mean                            
    projection = np.dot(mean_subtracted, eigenvectors.T)           
    return projection.flatten()

def calculate_similarity(proj1, proj2):
    if norm(proj1) == 0 or norm(proj2) == 0:
        return 0
    similarity = dot(proj1, proj2) / (norm(proj1) * norm(proj2))
    return round(similarity * 100, 2)

def get_all_faces_projections(images, mean, eigenvectors):
    return [project_face(face, mean, eigenvectors) for face in images]

def search_face(input_face, mean, eigenvectors, database_labels, db_projections, threshold):  
    input_proj = project_face(input_face, mean, eigenvectors)
    best_match_label = None
    best_match_similarity = 0
    for label, proj in zip(database_labels, db_projections):
        sim = calculate_similarity(input_proj, proj)
        if sim > best_match_similarity:
            best_match_similarity = sim
            best_match_label = label
        print(f"Comparing with {label}, similarity: {sim:.2f}%")
    if best_match_similarity >= threshold:
        return best_match_label, best_match_similarity
    else:
        return None, best_match_similarity

def resize_face():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open the camera!")
        return
    print("Camera is on. Press SPACE to capture face, ESC to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('Face Detection', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  
            break
        elif key == 32:  
            if len(faces) == 0:
                print("No faces detected.")
                continue
            (x, y, w, h) = faces[0]
            face_img = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (100, 100))
            cap.release()
            cv2.destroyAllWindows()
            return face_resized
    cap.release()
    cv2.destroyAllWindows()

def evaluate_system(face_resized, images, labels, max_components, threshold):
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)
    mean, eigenvectors = train_eigenfaces(X_train, max_components=max_components)
    train_proj = get_all_faces_projections(X_train, mean, eigenvectors)
    total = len(X_test)
    correct = 0
    false_positive = 0
    false_negative = 0
    total_time = 0

    for face, true_label in zip(X_test, y_test):
        start_time = time.time()
        pred_label, similarity = search_face(face_resized, mean, eigenvectors, y_train, train_proj, threshold)
        elapsed = time.time() - start_time
        total_time += elapsed

        if pred_label == true_label:
            correct += 1
        elif pred_label is None:
            false_negative += 1
        else:
            false_positive += 1

    accuracy = correct / total if total > 0 else 0
    avg_time = total_time / total if total > 0 else 0
    fpr = false_positive / total if total > 0 else 0
    fnr = false_negative / total if total > 0 else 0

    print("\n=== Evaluation Result ===")
    print(f"Max Component: {max_components}")
    print(f"Threshold: {threshold}")
    print(f"True label: {true_label}, Predicted: {pred_label}, Similarity: {similarity:.2f}%")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Average Matching Time: {avg_time:.4f} seconds")
    print(f"False Positive Rate: {fpr:.2f}")
    print(f"False Negative Rate: {fnr:.2f}")

def recognize_face(face_resized, mean, eigenvectors, database_labels, db_projections, threshold):
    label, similarity = search_face(face_resized, mean, eigenvectors, database_labels, db_projections, threshold)
    if label:
        print(f"Face recognized as {label} with similarity {similarity:.2f}%")
    else:
        print(f"Face not recognized. Similarity: {similarity:.2f}%")

def main():
    create_db()
    folder = 'img'
    images, labels = load_face_images_from_folder(folder)
    for img, label in zip(images, labels):
        img_hash = hash_image(img)
        insert_face(label, img_hash)
    if not images:
        print("No face images found.")
        return
    print(f"Loaded {len(images)} face images.")
    mean, eigenvectors = train_eigenfaces(images)
    db_projections = get_all_faces_projections(images, mean, eigenvectors)
    face_resized = resize_face()
    if face_resized is not None:
        recognize_face(face_resized, mean, eigenvectors, labels, db_projections, threshold=65)
    else:
        print("No face captured.")
    evaluate_system(face_resized, images, labels, max_components=10, threshold=65)

if __name__ == "__main__":
    main()