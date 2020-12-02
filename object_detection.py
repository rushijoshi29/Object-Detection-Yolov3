import cv2
import numpy as np

cap = cv2.VideoCapture(0)

whT = 320
confThreshold = 0.5
nmsThreshold = 0.3
classesFile = 'coco_names'
classNames = []

with open(classesFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)
model_Configuration = 'yolov3.cfg'
model_Weights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(model_Configuration,model_Weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img, n):

    hT,wT,cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT) - w/2) , int((det[1]*hT) - h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))


    #print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)

    for i in indices:

        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        n = classNames[classIds[i]]

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f'{n.upper()} {int(confs[i]*100)}%', (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
        #cv2.putText(img, n, (j+1,j), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255),2)
        #cv2.putText(img, n, (10,10), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (255,0,255), 2)

    return n

while True:

    success, img = cap.read()
    img_counter = 0
    pass_arg = []
    n = []

    cv2.imshow('Image', img)

    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    elif k%256 == 32:
        # SPACE pressed
        #print("Total Number of Tags : ")

        blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layerNames = net.getLayerNames()

        outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        outputs = net.forward(outputNames)
        n = findObjects(outputs, img, pass_arg)
        print(n)
        #cv2.putText(img, f'{"Total Number of Tags : "} {n}', (100, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, img)
        print("{} written!".format(img_name))
        img_counter += 1

cv2.imshow('Image', img)




