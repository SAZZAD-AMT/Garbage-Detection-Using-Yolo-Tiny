
import cv2

thres = 0.5 #Threshold to detect object

cap = cv2.VideoCapture(0)



classNames = []
classFiles='Garbage.csv'
with open(classFiles,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


weightsPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
configPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


print('\n Encoding Complete- Press Q or q to CLOSE WEBCAM')
while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    print(classIds,bbox)

    if len(classIds)!= 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0,255,0), thickness=3)
            cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            #cv2.putText(img,str(confidence),(box[0]+50,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    
    cv2.imshow("Output",img)
    b=cv2.waitKey(1)
    if b==31 or b==113:
        print("End Detection")
        break
    