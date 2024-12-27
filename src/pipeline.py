from deepface import DeepFace
from threading import Thread


def compute_embedding(img):
    return DeepFace.represent(img, detector_backend="skip", model_name="Facenet")


def detect(image):
    return DeepFace.extract_faces(
        image,
        detector_backend="yunet",
        enforce_detection=False,
        normalize_face=False,
        # color_face="bgr",
        expand_percentage=10,
    )


class DetectThread(Thread):
    def __init__(
        self, group=None, target=None, name=None, args=..., kwargs=None, *, daemon=None
    ):
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self.frame = None
        self.result = None

    def update(self):

        if self.frame is not None:
            self.result = detect(self.frame)

    def load_frame(self, frame):
        self.frame = frame
