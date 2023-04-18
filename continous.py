import cv2
import numpy as np

# Load YOLOv4 weights and configuration
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Load object class names
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set input size and scale factor
input_size = (416, 416)
scale_factor = 1/255.0

# Initialize variables for previous frame
prev_frame = None
prev_centroids1 = {}
prev_centroids2 = {}
# Start video capture
cap1 = cv2.VideoCapture('car_short.mp4')
#cap2 = cv2.VideoCapture(0)

cap2 = cv2.VideoCapture('car_short.mp4')
xval=0
speed1=0
speed2=0
speedx=-9999
speedy=-9999
yval=0
limit=10
while True:
    # Read frames from video captures
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break

    # Resize frames and convert to blobs
    resized1 = cv2.resize(frame1, input_size)
    blob1 = cv2.dnn.blobFromImage(resized1, scale_factor, input_size, swapRB=True)
    resized2 = cv2.resize(frame2, input_size)
    blob2 = cv2.dnn.blobFromImage(resized2, scale_factor, input_size, swapRB=True)

    # Pass blobs through YOLOv4 network
    net.setInput(blob1)
    outs1 = net.forward(net.getUnconnectedOutLayersNames())
    net.setInput(blob2)
    outs2 = net.forward(net.getUnconnectedOutLayersNames())

    # Initialize variables for current frame
    curr_centroids1 = {}
    curr_centroids2 = {}

    # Loop over detected objects in first stream
    for out in outs1:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Check if object is a vehicle
            if classes[class_id] == "car" or classes[class_id] == "truck" or classes[class_id] == "bus":
                # Calculate object bounding box coordinates
                box = detection[0:4] * np.array([frame1.shape[1], frame1.shape[0], frame1.shape[1], frame1.shape[0]])
                (x, y, w, h) = box.astype("int")

                # Calculate centroid of object
                cx = int(x + w/2)
                cy = int(y + h/2)
                centroid = (cx, cy)

                # Add centroid to dictionary
                curr_centroids1[centroid] = True

                # Draw bounding box and centroid on frame
                cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.circle(frame1, centroid, 5, (0, 255, 0), -1)

    # Loop over detected objects in second stream
    for out in outs2:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Check if object is a vehicle
            if classes[class_id] == "car" or classes[class_id] == "truck" or classes[class_id] == "bus":
                # Calculate object bounding box coordinates
                 box = detection[0:4] * np.array([frame2.shape[1], frame2.shape[0], frame2.shape[1], frame2.shape[0]])
                 (x, y, w, h) = box.astype("int")

                # Calculate centroid of object
                 cx = int(x + w/2)
                 cy = int(y + h/2)
                 centroid = (cx, cy)

                # Add centroid to dictionary
                 curr_centroids2[centroid] = True

                # Draw bounding box and centroid on frame
                 cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 2)
                 cv2.circle(frame2, centroid, 5, (0, 255, 0), -1)

    # Calculate speed of each object based on centroids in current and previous frames

    for centroid in curr_centroids1:
        if centroid in prev_centroids1:
            prev_centroid1 = prev_centroids1[centroid]
            distance = np.linalg.norm(np.array(centroid) - np.array(prev_centroid1))
            speed1 = distance / 5.2 # Assuming 1 second between frames
            #print("Object speed:1", speed)
            xval=xval+1
            if speedx<speed1:
                speedx=speed1
            if xval==limit:
                break

    # Store current centroids as previous centroids for next iteration
    prev_frame = frame2.copy()
    prev_centroids1 = curr_centroids1
    
    
    cv2.imshow("Video1", frame1)
    cv2.waitKey(1)

        # Calculate speed of each object based on centroids in current and previous frames
    
    for centroid in curr_centroids2:
        if centroid in prev_centroids2:
            prev_centroid2 = prev_centroids2[centroid]
            distance = np.linalg.norm(np.array(centroid) - np.array(prev_centroid2))
            speed2 = distance / 5.2 # Assuming 1 second between frames
            #print("Object speed:2", speed)
            if speedy<speed2:
                speedy=speed2
            yval=yval+1
            if y==limit:
                break

    # Store current centroids as previous centroids for next iteration
    prev_frame = frame2.copy()
    prev_centroids2 = curr_centroids2
    
    
    cv2.imshow("Video2", frame2)
    cv2.waitKey(1)
    if xval%limit==0 | yval%limit==0 :
        print("Object speed:1", speedx)
        print("Object speed:2", speedy)
        if (speedx!=-9999) & (speedy!=-9999):
            if (speedy>=50) | (speedx>=50):
               print("warning")
            else:
               print("safe")
        

# if (speedx!=-9999) & (speedy!=-9999):
#     if (speedy>=50) | (speedx>=50):
#         print("warning")
# else:
#     print("safe")