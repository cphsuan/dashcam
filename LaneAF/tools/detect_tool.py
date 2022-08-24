import cv2

def drawBoundingBox(img, bboxs):
    for box in bboxs:
        x1,y1,x2,y2 = (box['x1'], box['y1'], box['x2'], box['y2'])
        label = box['label']
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 6)
        fontFace = cv2.FONT_HERSHEY_COMPLEX
        fontScale = 0.5
        thickness = 1
        labelSize = cv2.getTextSize(label, fontFace, fontScale, thickness)
        _x1 = x1 # bottomleft x of text
        _y1 = y1 # bottomleft y of text
        _x2 = x1+labelSize[0][0] # topright x of text
        _y2 = y1-labelSize[0][1] # topright y of text
        cv2.rectangle(img, (_x1,_y1), (_x2,_y2), (0,255,0), cv2.FILLED) # text background
        cv2.putText(img, label, (x1,y1), fontFace, fontScale, (0,0,0), thickness)
    return img