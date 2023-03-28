import cv2
from playsound import playsound

#-----multi thread testing
import multiprocessing
import time
import os

from video_streaming import FileVideoStream


# Distance constants
KNOWN_DISTANCE_S = 0.99 # meters
KNOWN_DISTANCE_T = 2.24
KNOWN_DISTANCE_W = 2.51

SIGN_WIDTH = 0.60
TREE_WIDTH = 1.68
WIRES_WIDTH = 2.24




# Object detector constant
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3 #non max supression

# colors for object detected
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
# defining fonts
FONTS = cv2.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file
#yolo
class_names = []
with open("obj.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv2.dnn.readNet('yolov4-tiny-custom_best.weights', 'yolov4-tiny-custom.cfg')

#yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
#yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

net = cv2.dnn_DetectionModel(yoloNet)
net.setInputParams(size=(416, 416), scale=1/255, swapRB=True)





# object detector funciton /method
def object_detector(image):
    start_time = time.time()
    classes, scores, boxes = net.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    print("model prediction: ", time.time() - start_time)
    # creating empty list to add objects data
    data_list = []

    global box
    global i,j,w,h
    
    for (classid, score, box) in zip(classes, scores, boxes):
        
        #print(classid, score)
        i, j, w, h = box

        # define color of each, object based on its class id
        color = COLORS[int(classid) % len(COLORS)]

        label = "%s : %f" % (class_names[classid], score)

        # getting the data
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid == 1:  # stop sign class id
            data_list.append(["sign", box[2], (box[0], box[1] - 2)])

            #cv2.rectangle(image, box, color, 2)
            #cv2.putText(image, label, (box[0], box[1] - 14), FONTS, 0.5, color, 2)

        elif classid == 2:
            data_list.append(["tree", box[2], (box[0], box[1] - 2)])

            cv2.rectangle(image, box, color, 2)
            cv2.putText(image, label, (box[0], box[1] - 14), FONTS, 0.5, color, 2)

        elif classid == 3:
            data_list.append(["wires", box[2], (box[0], box[1] - 2)])

            cv2.rectangle(image, box, color, 2)
            cv2.putText(image, label, (box[0], box[1] - 14), FONTS, 0.5, color, 2)


        # if you want to include more classes then you have to simply add more [elif] statements here
        # returning list containing the object data.
        print("object detection: ", time.time()-start_time)
    return data_list




def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length / 7



# distance finder function
def distance_finder(focal_length, real_object_width, width_in_frmae):
    global distance
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance

#cap = cv2.VideoCapture(-1)

# reading the reference image from dir
ref_sign = cv2.imread('IMG_7142.jpg')
ref_tree = cv2.imread('tree.JPG')
ref_wires = cv2.imread('wires_1.JPG')

sign_data = object_detector(ref_sign)
sign_width_in_rf = sign_data[0][1]
tree_data = object_detector(ref_tree)
tree_width_in_rf = tree_data[0][1]
wires_data = object_detector(ref_wires)
wires_width_in_rf = wires_data[0][1]

# finding focal length
focal_tree = focal_length_finder(KNOWN_DISTANCE_T, TREE_WIDTH, tree_width_in_rf)
focal_wires = focal_length_finder(KNOWN_DISTANCE_W, WIRES_WIDTH, wires_width_in_rf)
focal_sign = focal_length_finder(KNOWN_DISTANCE_S, SIGN_WIDTH, sign_width_in_rf)


notif_count = 0
consecutive_safe_counter = 0
#tracker.init(frame, box)

def sound_player(filename = ""):
    # os.system('afplay /home/pi/Computer-Vision-for-Visuallt-Impaired/'+filename+'.mp3 &')
    sound_root = "/home/pi/Computer-Vision-for-Visuallt-Impaired"
    fname = "{}.wav".format(filename)
    fpath = os.path.join(sound_root, fname)
    playsound(fpath)

frame_queue = FileVideoStream(-1).start()


while True:
    #ret, frame = cap.read()
    frame = frame_queue.read()
    
    data = object_detector(frame)
    #cv2.waitKey(50)
    if len(data) > 0:
        consecutive_safe_counter = 0
        for d in data:
            objectName = d[0]

            if objectName == 'sign':
                
                x, y = d[2]  # position where to draw text, dist
                distance = distance_finder(focal_sign, SIGN_WIDTH, d[1])

                #cv2.rectangle(frame, (x, y - 3), (x + 150, y + 23), BLACK, -1)
                #cv2.putText(frame, f'Dis: {round(distance, 2)} m', (x + 5, y + 13), FONTS, 0.48, GREEN, 2)

                #img capped from webcam dimensions: (720, 1280, 3)
                if distance < 15:
                    #if success:
                        #if notif_count == 0:
                    #print("i: {}, w: {}, (i+w)*2: {}".format(i, w, (i+w)*2))
                    if (i+w)*2 in range(0,640): #left-half region
                        # print('Move right')
                        # playsound('Move_right.mp3', False)
                        # ------ new way to play sound
                        print('Beware your left')
                        if notif_count%30 == 0:
                            sound_player("move_left")
                        # ------end  new way to play sound
                        #time.sleep(w_time)  # delay for 3 secs
                        notif_count += 1

                    elif (i+w)*2 in range(641,1280): #right-half region
                        # print('Move left')
                        # playsound('Move_left.mp3', False)
                        # ------ new way to play sound
                        print('Beware your right')
                        if notif_count%30 == 0:
                            notif_count = 0
                            sound_player("move_right")
                        # ------end  new way to play sound
                        #time.sleep(w_time)  # delay for 3 secs
                        notif_count += 1

    else:
        consecutive_safe_counter += 1
        if notif_count > 0:
            if consecutive_safe_counter >= 1:
                consecutive_safe_counter = 0
                print('Safe')
                sound_player("safe")
                
                notif_count = 0

    #cv2.imshow('frame', frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    

cv2.destroyAllWindows()
cap.release()