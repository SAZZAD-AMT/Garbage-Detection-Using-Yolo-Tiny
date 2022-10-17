
import cv2
import numpy as np

cap = cv2.VideoCapture(1)
whT=320

classNames = []
classFiles='coco.names'
with open(classFiles,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

weightsPath = "yolov3-tiny.weights"
configPath = "yolov3-tiny.cfg"

net = cv2.dnn.readNetFromDarknet(weightsPath,configPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)



print('\n Encoding Complete- Press Q or q to CLOSE WEBCAM')
while True:
    success,img = cap.read()
    blob=cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames=net.getLayerNames()
    #print(layerNames)
    outputNames=[layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    #print(net.getUnconnectedOutLayers())
    outputs=net.forward(outputNames)
    print(outputs[0].shape)
    print(outputs[1].shape)
    print(outputs[2].shape)


    
    cv2.imshow("Image",img)
    b=cv2.waitKey(1)
    if b==31 or b==113:
        print("End Detection")
        break
    

    