import cv2
from playsound import playsound
import time


# Distance constants
KNOWN_DISTANCE = 45  # INCHES
PERSON_WIDTH = 16  # INCHES
MOBILE_WIDTH = 3.0  # INCHES


KNOWN_DISTANCE_S = 5
SIGN_WIDTH = 23.5

# Object detector constant
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3 #non max supression

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
    global box
    global i,j,w,h

    for (classid, score, box) in zip(classes, scores, boxes):
        i, j, w, h = box

        # define color of each, object based on its class id
        color = COLORS[int(classid) % len(COLORS)]

        label = "%s : %f" % (class_names[classid - 1], score)

        # draw rectangle on and label on object
        cv2.rectangle(image, box, color, 2)
        cv2.putText(image, label, (box[0], box[1] - 14), FONTS, 0.5, color, 2)




        # getting the data
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid == 13:  # stop sign class id
            # x, y, w, h = box[0], box[1], box[2], box[3]
            data_list.append(["stop sign", box[2], (box[0], box[1] - 2)])

        elif classid == 77:
            data_list.append(["cell phone", box[2], (box[0], box[1] - 2)])


        #elif classid == 12:
            #data_list.append(["street sign", box[2], (box[0], box[1] - 2)])
        # if you want to include more classes then you have to simply add more [elif] statements here
        # returning list containing the object data.
    return data_list


def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length



# distance finder function
def distance_finder(focal_length, real_object_width, width_in_frmae):
    global distance
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance




#tracker
cap = cv2.VideoCapture(0)
tracker = cv2.legacy.TrackerMOSSE_create() #high speed tracker
ret, frame = cap.read()
onTracking = False




# reading the reference image from dir
ref_sign = cv2.imread('stop_sign.jpeg')
ref_mobile = cv2.imread('image4.png')
ref_person = cv2.imread('image14.png')


sign_data = object_detector(ref_sign)
sign_width_in_rf = sign_data[0][1] #first position 'if', box[2]

mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[1][1] #second position 'elif', box[2]







# finding focal length
#focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)

focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)

focal_sign = focal_length_finder(KNOWN_DISTANCE_S, SIGN_WIDTH, sign_width_in_rf)




cap = cv2.VideoCapture(0)

notif_count = 0
#tracker.init(frame, box)


while True:
    ret, frame = cap.read()
    data = object_detector(frame)
    tracker.init(frame, box)
    success, box = tracker.update(frame)

    #data = object_detector(frame)


    for d in data:
        if d[0] == 'stop sign':
            distance = distance_finder(focal_sign, SIGN_WIDTH, d[1])
            x, y = d[2] #position where to draw text, dist
            #img capped from webcam dimensions: (720, 1280, 3)
            if distance < 20:
                if success:
                    #if notif_count == 0:
                    if (i+w) in range(0,640): #left-half region
                        print('Move right')
                        playsound('Move_right.mp3')
                        #time.sleep(w_time)  # delay for 3 secs
                        notif_count += 1

                    elif (i+w) in range(641,1280): #right-half region
                        print('Move left')
                        playsound('Move_left.mp3')
                        #time.sleep(w_time)  # delay for 3 secs
                        notif_count += 1

                else:
                    print('Safe')
                    playsound('Safe.mp3')
                    print(box)
                    notif_count = 0
                    print(i+w)


        elif d[0] == 'cell phone':
            distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]





            print(box)
            print(i+w)


        cv2.rectangle(frame, (x, y - 3), (x + 150, y + 23), BLACK, -1)
        cv2.putText(frame, f'Dis: {round(distance, 2)} inch', (x + 5, y + 13), FONTS, 0.48, GREEN, 2)


    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()