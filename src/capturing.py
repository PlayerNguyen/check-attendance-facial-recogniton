import os
import cv2
from src.pipeline import detect, DetectThread
from src.asset import PARENT_DIRECTORY


def ensure_parent():
    os.makedirs(PARENT_DIRECTORY, exist_ok=True)


def capturing_mode():
    name = input("Enter a name: ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    ensure_parent()

    directory_path = os.path.join(os.getcwd(), ".data", name)

    try:

        os.makedirs(directory_path, exist_ok=True)
        print(f"Directory '{directory_path}' created successfully.")
        start_capture(name)
    except Exception as e:
        print(f"Error creating directory: {e}")


def start_capture(name):

    cap = cv2.VideoCapture(0)
    detect_thread = DetectThread()
    last_result = ""
    last_bboxs = []
    gap = 0
    count = 0
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

            # Not found any face
            if len(filtered_data) == 0:
                last_bboxs = []
                last_result = "Cannot found any face"
            elif len(filtered_data) > 1:
                last_result = "Capturing mode does not accept multiple faces."
            else:
                # Draw bounding box
                last_result = ""
                facial = filtered_data[0]["facial_area"]
                x = facial["x"]
                y = facial["y"]
                w = facial["w"]
                h = facial["h"]
                last_bboxs = [[x, y, w, h]]

                print(filtered_data[0]["face"])
                # Embedding it
                if count >= 1:
                    break
                else:
                    save_path = os.path.join(PARENT_DIRECTORY, name, f"{count}.png")
                    cv2.imwrite(save_path, filtered_data[0]["face"])
                    count += 1
                    print(
                        f"Successfully captured the face. The face is stored into database now."
                    )

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
        for bbox in last_bboxs:
            bbox = list(map(lambda a: int(a), bbox))
            x, y, w, h = bbox
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 4)

        # Clean up before display
        cv2.imshow("Preview", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
