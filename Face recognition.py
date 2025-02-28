import cv2
import face_recognition
import os
import glob
import numpy as np


class FaceRecognition:
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.frame_resizing = 0.25  # Resize frame for efficiency

    def load_images(self, images_path):
        """
        Load images and generate encodings
        """
        images = glob.glob(os.path.join(images_path, "*.*"))
        print(f"{len(images)} images found for encoding.")

        for img_path in images:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            basename = os.path.basename(img_path)
            filename, _ = os.path.splitext(basename)

            try:
                encoding = face_recognition.face_encodings(rgb_img)[0]
                self.known_encodings.append(encoding)
                self.known_names.append(filename)
            except IndexError:
                print(f"No face detected in {filename}, skipping.")

        print("Encoding complete.")

    def recognize_faces(self, frame):
        """
        Detect and identify known faces in the frame
        """
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_encodings, encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(self.known_encodings, encoding)
            best_match = np.argmin(face_distances)
            if matches[best_match]:
                name = self.known_names[best_match]
            face_names.append(name)

        return np.array(face_locations) // self.frame_resizing, face_names


if __name__ == "__main__":
    fr = FaceRecognition()
    fr.load_images("images")

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        locations, names = fr.recognize_faces(frame)
        for (top, right, bottom, left), name in zip(locations, names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
