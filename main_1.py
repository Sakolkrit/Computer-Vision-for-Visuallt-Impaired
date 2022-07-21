import cv2
import pyttsx3
import numpy as np

engine = pyttsx3.init()

def blind_speak(command):
    engine.say(command)
    engine.runAndWait()

# Distance constants
KNOWN_DISTANCE = 45  # INCHES
PERSON_WIDTH = 16  # INCHES
MOBILE_WIDTH = 3.0  # INCHES

# Object detector constant
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
# defining fonts
FONTS = cv2.FONT_HERSHEY_COMPLEX

class_names = [] #type straight up
classFile = 'coco.names'

with open(classFile, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

#net
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)



# object detector funciton /method
def object_detector(image):
    classes, scores, boxes = net.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id
        color = COLORS[int(classid) % len(COLORS)]

        #label = "%s : %f" % (class_names[classid[0]], score)
        label = "person"

        # draw rectangle on and label on object
        cv2.rectangle(image, box, color, 2)
        cv2.putText(image, label, (box[0], box[1] - 14), FONTS, 0.5, color, 2)

        # getting the data
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid == 1:  # person class id
            #data_list.append([class_names[classid[0]], box[2], (box[0], box[1] - 2)])
            data_list.append(["person", box[2], (box[0], box[1] - 2)])
        elif classid == 68:
            #data_list.append([class_names[classid[0]], box[2], (box[0], box[1] - 2)])
            data_list.append(["window", box[2], (box[0], box[1] - 2)])
        # if you want include more classes then you have to simply add more [elif] statements here
        # returning list containing the object data.
    return data_list


def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length


# distance finder function
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance


# reading the reference image from dir
ref_person = cv2.imread('image14.png')
ref_mobile = cv2.imread('image4.png')

mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[0][1]

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

print(f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf}")

# finding focal length
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)

focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)
cap = cv2.VideoCapture(0)

notif_count = 0

while True:
    ret, frame = cap.read()

    data = object_detector(frame)
    for d in data:
        if d[0] == 'person':
            distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
            x, y = d[2]
            if notif_count == 0:
                if distance <= 20: #first time seeing object, distance < 20 in
                    blind_speak("Person detected. Continue moving.")
                    notif_count += 1
                    if notif_count > 0: #object seen before
                        #if distance <=20:
                        blind_speak('Safe.')
                        notif_count = 0


        else:
            pass #to next frame

        #elif d[0] == 'cell phone':
            #distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
            #x, y = d[2]
        cv2.rectangle(frame, (x, y - 3), (x + 150, y + 23), BLACK, -1)
        cv2.putText(frame, f'Dis: {round(distance, 2)} inch', (x + 5, y + 13), FONTS, 0.48, GREEN, 2)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()