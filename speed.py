import cv2
import numpy as np

net = cv2.dnn.readNet("C:/Users/sonia/Desktop/speed/yolov4.weights", "C:/Users/sonia/Desktop/speed/yolov4.cfg")


classes = []
with open("C:/Users/sonia/Desktop/speed/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set input size and scale factor
#input_size = (608, 608)
input_size = (416, 416)
scale_factor = 1/255.0

# Initialize variables for previous frame
prev_frame = None
prev_centroids = {}

# Start video capture
cap = cv2.VideoCapture('car_short.mp4')
# cap = cv2.VideoCapture(0)
count=0
while True:
    # Read frame from video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame and convert to blob
    resized = cv2.resize(frame, input_size)
    blob = cv2.dnn.blobFromImage(resized, scale_factor, input_size, swapRB=True)

    # Pass blob through YOLOv4 network
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Initialize variables for current frame
    curr_centroids = {}

    # Loop over detected objects
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Check if object is a vehicle
            if classes[class_id] == "car" or classes[class_id] == "truck" or classes[class_id] == "bus":
                # Calculate object bounding box coordinates
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (x, y, w, h) = box.astype("int")

                # Calculate centroid of object
                cx = int(x + w/2)
                cy = int(y + h/2)
                centroid = (cx, cy)

                # Add centroid to dictionary
                curr_centroids[centroid] = True

                # Draw bounding box and centroid on frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.circle(frame, centroid, 5, (0, 255, 0), -1)

    # Calculate speed of each object based on centroids in current and previous frames

    
    for centroid in curr_centroids:
        if centroid in prev_centroids:
            
            prev_centroid = prev_centroids[centroid]
            distance = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))
            speed = distance / 1.0 # Assuming 1 second between frames
            print("Object speed: "+str(count) +" -> " + str(speed))
            count=count+1


    # Store current centroids as previous centroids for next iteration
    prev_frame = frame.copy()
    prev_centroids = curr_centroids
    
    
    cv2.imshow("Video", frame)
    cv2.waitKey(10000)
    

cap.release()
cv2.destroyAllWindows()
