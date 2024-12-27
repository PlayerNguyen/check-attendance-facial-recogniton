import os
import cv2
from src.pipeline import compute_embedding, DetectThread
from src.asset import PARENT_DIRECTORY
from numpy import dot
from numpy.linalg import norm
import torch
import numpy as np
from src.attendance import check_attendance

data = []
THRESHOLD = 0.6


def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def find_similarity(embedding_vector):

    only_embedding_from_db = [x[1] for x in data]
    similarities = [
        cosine_similarity(embedding, embedding_vector)
        for embedding in only_embedding_from_db
    ]
    return similarities


def load_memory():
    print("Indexing data into memory...")
    for dir in os.listdir(PARENT_DIRECTORY):
        image_path = os.path.join(PARENT_DIRECTORY, dir, "0.png")
        image = cv2.imread(image_path)
        embedding = compute_embedding(image)[0]["embedding"]
        name = dir
        data.append([name, embedding])
    print(f"Data is loaded into the memory. Found {len(data)} profile(s).")


def start_inference():
    cap = cv2.VideoCapture(0)
    detect_thread = DetectThread()
    last_result = ""
    last_bboxs = []
    last_labels = []
    gap = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            raise "Cannot read the video capture"

        detect_thread.load_frame(frame)
        gap += 1

        if gap == 10:
            gap = 0
            detect_thread.update()

            # Draw bonding box on a face with confidence greater than zero
            # print(detect_thread.result)
            filtered_data = list(
                filter(lambda a: a["confidence"] > 0, detect_thread.result)
            )

            if len(filtered_data) != 0:
                # Draw bounding box
                last_result = ""
                facial = filtered_data[0]["facial_area"]
                face = filtered_data[0]["face"]
                x = facial["x"]
                y = facial["y"]
                w = facial["w"]
                h = facial["h"]
                last_bboxs = [[x, y, w, h]]

                # Compute embedding for each images
                face_embedding = compute_embedding(face)[0]["embedding"]
                similarities = find_similarity(face_embedding)
                idx = np.argmax(similarities)
                label, embedding = data[idx]
                similarity = similarities[idx]
                last_labels = [[label, similarity]]
                print(similarity)
                # Write out to the file
                if similarity >= THRESHOLD:
                    check_attendance(label)

            else:
                last_bboxs = []

        frame = cv2.putText(
            frame,
            last_result,
            (15, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
        )

        # Draw bbox
        for bbox, label in zip(last_bboxs, last_labels):
            bbox = list(map(lambda a: int(a), bbox))
            x, y, w, h = bbox
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 4)
            # draw similiarity
            frame = cv2.putText(
                frame,
                f"{label[0]} ({label[1]})",
                (x + 5, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                1,
            )

        # Clean up before display
        cv2.imshow("Preview recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
