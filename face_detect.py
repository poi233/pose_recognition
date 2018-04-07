import face_recognition


# 识别图片unknown_image是否在face_point附近能找到人脸known_image
def recognize_person(known_image_path, unknown_image, face_point):
    known_image = face_recognition.load_image_file(known_image_path)
    known_encoding = face_recognition.face_encodings(known_image)[0]
    # get a list of tuples of found face locations in css (top, right, bottom, left) order
    face_locations = face_recognition.face_locations(unknown_image)
    shape = unknown_image.shape
    for location in face_locations:
        y_start = location[0]-10 if location[0] - 10 >= 0 else location[0]
        y_end = location[2]+10 if location[2] + 10 <= shape[0] else location[2]
        x_start = location[3]-10 if location[3] - 10 >= 0 else location[3]
        x_end = location[1]+10 if location[1] + 10 <= shape[1] else location[1]
        _face_point = (int(face_point[1] * shape[0]), int(face_point[0] * shape[1]))
        if y_start < _face_point[0] < y_end and x_start < _face_point[1] < x_end:
            unknown_face = unknown_image[y_start:y_end, x_start:x_end]
            unknown_encoding = face_recognition.face_encodings(unknown_face)
            if not unknown_encoding:
                return False
            else:
                result = face_recognition.compare_faces([known_encoding], unknown_encoding[0])[0]
                return result
    return False

# recognize_person()