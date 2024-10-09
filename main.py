# pip install face_recognition
# pip install opencv-python
# pip install cmake
# pip install pyttsx3
import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()

video_capture = cv2.VideoCapture(0)

# Load Known faces and convert to RGB using OpenCV


marks_image = cv2.imread("faces/Mark.jpeg")
marks_image = cv2.cvtColor(marks_image, cv2.COLOR_BGR2RGB)
mark_encoding = face_recognition.face_encodings(marks_image)[0]

dwight_image=cv2.imread("faces/Dwight.jpeg")
dwight_image=cv2.cvtColor(dwight_image, cv2.COLOR_BGR2RGB)
dwight_encoding=face_recognition.face_encodings(dwight_image)[0]

jim_image=cv2.imread("faces/Jim.jpg")
jim_image=cv2.cvtColor(jim_image,cv2.COLOR_BGR2RGB)
jim_encoding=face_recognition.face_encodings(jim_image)[0]

michael_image=cv2.imread("faces/MichaelScott.jpg")
michael_image=cv2.cvtColor(michael_image,cv2.COLOR_BGR2RGB)
michael_encoding=face_recognition.face_encodings(michael_image)[0]

# Store known face encodings
known_face_encodings = [ mark_encoding,michael_encoding,jim_encoding,dwight_encoding]
known_face_names = [ "Mark","Michael","Jim","Dwight"]

# List of expected students
students = known_face_names.copy()

face_locations = []
face_encodings = []

# Get the current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Open CSV file for logging attendance
f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)
lnwriter.writerow(["Name", "Time"])

# Create a set to track logged and spoken students
logged_students = set()
spoken_students = set()

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        # Compare faces with known encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")

            if name in known_face_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 100)
                fontScale = 1.5
                fontColor = (0, 0, 255)
                thickness = 3
                lineType = 2
                cv2.putText(frame, name + " is here", (10, 100), font, fontScale, font)

                # Log attendance only once for each student
                if name not in logged_students:
                    lnwriter.writerow([name, current_time])
                    logged_students.add(name)
                print(f"{name} logged at {current_time}")

                # Speak welcome message only once
                if name not in spoken_students:
                    print(f"Hello, {name}!")
                    engine.say(f"Hello, {name}!")
                    engine.runAndWait()
                    spoken_students.add(name)

    cv2.imshow("Attendance", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
f.close()
