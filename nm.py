import face_recognition as fr
import cv2 as cv
import os
import numpy as np

https://anotepad.com/notes/2ragjbir
""" image_path = "/workspaces/codespaces-jupyter/images/dhoni/B_3.jpg"
target_image = fr.load_image_file(image_path)
target_encoding = fr.face_encodings(target_image)
 """

video_capture = cv.VideoCapture(0)


def encode_faces(folders):
    list_people_encoding = []
    known_face_encoding = []
    known_face_name = []
    for filename in os.listdir(folders):
        known_image = fr.load_image_file(f"{folders}{filename}")
        known_encoding = fr.face_encodings(known_image)[0]
        known_face_encoding.append(known_encoding)
        known_face_name.append(filename)

    return known_face_encoding , known_face_name

def create_frame(location,label,frame):
    print(location)
    top , right , bottom , left = location
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4

    cv.rectangle(frame , (left,top ),(right,bottom), (255,0,0),2)
    cv.rectangle(frame , (left,bottom+20),(right,bottom), (255,0,0),cv.FILLED)
    cv.putText(frame ,label,(left+3,bottom+14),cv.FONT_HERSHEY_DUPLEX,0.4,(255,255,255),1)

known_face_encodings , known_face_names = encode_faces("people/")


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret,frame=video_capture.read()

    if process_this_frame:
        small_frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = small_frame[:, :, ::-1]
        
        face_locations = fr.face_locations(rgb_small_frame)
        face_encodings = fr.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            
            matches = fr.compare_faces(known_face_encodings, face_encoding,tolerance=0.48)
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = fr.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    #create_frame(face_locations,face_names,frame)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv.FILLED)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv.destroyAllWindows()
