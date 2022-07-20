import cv2
import pyttsx3

engine = pyttsx3.init()

def blind_speak(command):
    engine.say(command)
    engine.runAndWait()

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
classNames = []
classFile = 'coco.names'

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

#model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

notif_count = 0

while True:
    ret, img = cap.read()

    #check webcam chosen
    if img is None:
        print("Wrong webcam chosen.")
    else:
        classIds, confs, bbox = net.detect(img, confThreshold = 0.5) #class_id, conferences, bounding_boxes

        try:
            if len(classIds !=0): #find object in label list
                for classID, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    try:
                        label = classNames[classID - 1]
                        confidence = round(confidence, 2)
                        if label == 'person' and confidence > 0.50:
                            cv2.rectangle(img, box, color= (0,255,0), thickness=2) #box style
                            cv2.putText(img, label, (box[0]+10, box[1]+30), #text style
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                            print(label + ": " + str(confidence))
                            if notif_count == 0: #first time a particular object is detected
                                blind_speak("Person detected. Continue moving.")
                                notif_count += 1 #not going to say again
                        elif notif_count > 0:
                            blind_speak('Safe.')
                            notif_count = 0
                    except IndexError:
                        pass #pass to next frame
            else:
                pass
        except TypeError:
            pass

        #ret, img = cap.read()
        cv2.imshow('Output',img)
        cv2.waitKey(1)



        #cv2.imshow('Output',img)
        #if cv2.waitKey(1) == ord('q'):
            #break

