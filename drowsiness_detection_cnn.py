"""
Drowsiness / Eye Blink Detection using a CNN

This single-file project contains:
- data loading helpers (expects a folder with subfolders 'open' and 'closed')
- a compact CNN model (Keras / TensorFlow)
- training script with augmentation and callbacks
- evaluation function
- realtime webcam demo that uses Haar cascades to detect face/eyes and predicts open/closed

Usage (examples):
# Prepare dataset: manually collect eye crops into dataset/open and dataset/closed
# Train model:
#   python drowsiness_detection_cnn.py --mode train --data_dir ./dataset --epochs 30 --batch 32 --out model.h5
# Evaluate model:
#   python drowsiness_detection_cnn.py --mode evaluate --data_dir ./dataset --model model.h5
# Run realtime detection (webcam):
#   python drowsiness_detection_cnn.py --mode realtime --model model.h5

Requirements (pip):
# tensorflow (>=2.x), opencv-python, numpy, matplotlib, imutils
# optional: playsound (for alarm), scikit-learn

Notes / Dataset suggestions:
- You can use the Closed Eyes in the Wild (CEW) dataset or create your own by cropping eye regions
- Expected directory layout for training/eval data:
# dataset/
#   open/
#     img001.jpg
#   closed/
#     img001.jpg

"""

import os
import argparse
import random
import time
from datetime import datetime

import numpy as np
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------- Config -----------------------
IMG_SIZE = (64, 64)  # input size for CNN
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ----------------------- Model -----------------------

def build_cnn(input_shape=(64,64,3)):
    model = Sequential()

    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ----------------------- Data helpers -----------------------

def load_images_from_folder(folder, label, img_size=IMG_SIZE):
    X = []
    y = []
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for f in files:
        try:
            img = load_img(f, target_size=img_size)
            arr = img_to_array(img)
            X.append(arr)
            y.append(label)
        except Exception as e:
            print(f"Skipping {f}: {e}")
    return X, y


def load_dataset(data_dir):
    open_dir = os.path.join(data_dir, 'open')
    closed_dir = os.path.join(data_dir, 'closed')
    X_open, y_open = load_images_from_folder(open_dir, 0)
    X_closed, y_closed = load_images_from_folder(closed_dir, 1)

    X = np.array(X_open + X_closed, dtype='float32') / 255.0
    y = np.array(y_open + y_closed)

    # one-hot encode
    y_cat = np.zeros((len(y), 2), dtype='float32')
    y_cat[np.arange(len(y)), y] = 1.0

    return X, y_cat

# ----------------------- Training -----------------------

def train_model(data_dir, out_path, epochs=30, batch_size=32):
    print('Loading dataset from', data_dir)
    X, y = load_dataset(data_dir)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y)

    # Augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    model = build_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    model.summary()

    callbacks = [
        ModelCheckpoint(out_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
    ]

    steps_per_epoch = max(1, len(X_train) // batch_size)

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_val, y_val),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    print('Training finished. Best model saved to', out_path)
    return history

# ----------------------- Evaluation -----------------------

def evaluate_model(data_dir, model_path):
    model = load_model(model_path)
    X, y = load_dataset(data_dir)
    preds = model.predict(X, batch_size=32)
    y_pred = np.argmax(preds, axis=1)
    y_true = np.argmax(y, axis=1)

    print('\nClassification report:')
    print(classification_report(y_true, y_pred, target_names=['open','closed']))
    print('\nConfusion matrix:')
    print(confusion_matrix(y_true, y_pred))

# ----------------------- Realtime demo -----------------------

def realtime_demo(model_path, alarm_sound_path=None, threshold_consecutive_closed=12, use_cuda=False):
    # threshold_consecutive_closed: number of consecutive frames classified as 'closed' to trigger alarm
    model = load_model(model_path)
    cascade_path = cv2.data.haarcascades
    face_cascade = cv2.CascadeClassifier(os.path.join(cascade_path, 'haarcascade_frontalface_default.xml'))
    eye_cascade = cv2.CascadeClassifier(os.path.join(cascade_path, 'haarcascade_eye.xml'))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Could not open webcam.')
        return

    closed_count = 0
    last_alarm_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(80,80))

        status_text = 'Unknown'
        for (x, y, w, h) in faces:
            # focus on upper half of face for eyes region
            roi_gray = gray[y:y+int(h/2), x:x+w]
            roi_color = frame[y:y+int(h/2), x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20,20))

            # We'll take the largest detected eye region(s) and run prediction per-eye
            eye_imgs = []
            for (ex, ey, ew, eh) in eyes:
                eye_roi_color = roi_color[ey:ey+eh, ex:ex+ew]
                try:
                    resized = cv2.resize(eye_roi_color, IMG_SIZE)
                    arr = resized.astype('float32') / 255.0
                    eye_imgs.append(arr)
                    # draw rectangle
                    cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0,255,0), 1)
                except Exception:
                    continue

            if len(eye_imgs) > 0:
                X_eye = np.array(eye_imgs)
                preds = model.predict(X_eye)
                # if any eye predicted closed -> treat as closed
                preds_labels = np.argmax(preds, axis=1)
                if np.mean(preds_labels) > 0.5:  # more closed than open
                    closed_count += 1
                    status_text = 'Closed'
                else:
                    closed_count = max(0, closed_count-1)
                    status_text = 'Open'

            # only process first face
            break

        # decide alarm
        if closed_count >= threshold_consecutive_closed:
            status = 'DROWSY'
            # play alarm (if provided and not played too recently)
            now = time.time()
            if alarm_sound_path and now - last_alarm_time > 5:
                try:
                    # playsound is blocking; optional
                    from playsound import playsound
                    playsound(alarm_sound_path, block=False)
                except Exception as e:
                    print('Could not play sound:', e)
                last_alarm_time = now
        else:
            status = 'OK'

        # overlay
        cv2.putText(frame, f'Status: {status}  [{status_text}]', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255) if status=='DROWSY' else (0,255,0), 2)
        cv2.imshow('Drowsiness Detection (press q to quit)', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------- Utilities -----------------------

def prepare_sample_dataset_from_video(video_path, output_dir, frame_step=5, eye_detector='haar'):
    """
    OPTIONAL helper: go through a face-video, detect eyes, and save crops to output_dir/open or closed
    This is a very rough helper and requires manual labeling afterwards. Not intended for production.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    cascade_path = cv2.data.haarcascades
    face_cascade = cv2.CascadeClassifier(os.path.join(cascade_path, 'haarcascade_frontalface_default.xml'))
    eye_cascade = cv2.CascadeClassifier(os.path.join(cascade_path, 'haarcascade_eye.xml'))

    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(80,80))
            for (x,y,w,h) in faces:
                roi_gray = gray[y:y+int(h/2), x:x+w]
                roi_color = frame[y:y+int(h/2), x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20,20))
                for (ex,ey,ew,eh) in eyes:
                    eye_roi_color = roi_color[ey:ey+eh, ex:ex+ew]
                    try:
                        os.makedirs(os.path.join(output_dir, 'crops'), exist_ok=True)
                        fn = os.path.join(output_dir, 'crops', f'crop_{saved:06d}.jpg')
                        cv2.imwrite(fn, eye_roi_color)
                        saved += 1
                    except Exception:
                        pass
        idx += 1
    cap.release()
    print('Saved', saved, 'eye crops to', os.path.join(output_dir, 'crops'))

# ----------------------- CLI -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train','evaluate','realtime','prepare'], required=True)
    parser.add_argument('--data_dir', type=str, help='Dataset directory containing open/ and closed/ subfolders')
    parser.add_argument('--model', type=str, help='Path to model file (.h5)')
    parser.add_argument('--out', type=str, default='drowsiness_model.h5', help='Output path to save model')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--alarm', type=str, help='Path to alarm sound (wav/mp3) for realtime mode')
    parser.add_argument('--video', type=str, help='Source video for prepare mode')
    args = parser.parse_args()

    if args.mode == 'train':
        if not args.data_dir:
            print('Please provide --data_dir')
            return
        train_model(args.data_dir, args.out, epochs=args.epochs, batch_size=args.batch)

    elif args.mode == 'evaluate':
        if not args.data_dir or not args.model:
            print('Please provide --data_dir and --model')
            return
        evaluate_model(args.data_dir, args.model)

    elif args.mode == 'realtime':
        if not args.model:
            print('Please provide --model')
            return
        realtime_demo(args.model, alarm_sound_path=args.alarm)

    elif args.mode == 'prepare':
        if not args.video or not args.data_dir:
            print('Please provide --video and --data_dir')
            return
        prepare_sample_dataset_from_video(args.video, args.data_dir)

if __name__ == '__main__':
    main()
