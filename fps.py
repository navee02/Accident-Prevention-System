# import cv2
# import numpy as np
# import time

# # Load YOLOv4 weights and configuration
# net = cv2.dnn.readNet("C:/Users/sonia/Desktop/speed/yolov4.weights", "C:/Users/sonia/Desktop/speed/yolov4.cfg")

# # Load object class names
# classes = []
# with open("C:/Users/sonia/Desktop/speed/coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]

# # Set input size and scale factor
# input_size = (608, 608)
# scale_factor = 1/255.0

# # Initialize variables for previous frame
# prev_frame = None
# prev_centroids = {}

# # Start video capture
# cap = cv2.VideoCapture('car_short.mp4')
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# # Initialize variables for FPS calculation
# fps_start_time = None
# fps_frame_count = 0
# fps = 0

# while True:
#     # Read frame from video capture
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Resize frame and convert to blob
#     resized = cv2.resize(frame, input_size)
#     blob = cv2.dnn.blobFromImage(resized, scale_factor, input_size, swapRB=True)

#     # Pass blob through YOLOv4 network
#     net.setInput(blob)
#     outs = net.forward(net.getUnconnectedOutLayersNames())

#     # Initialize variables for current frame
#     curr_centroids = {}

#     # Loop over detected objects
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]

#             # Check if object is a vehicle
#             if classes[class_id] == "car" or classes[class_id] == "truck" or classes[class_id] == "bus":
#                 # Calculate object bounding box coordinates
#                 box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
#                 (x, y, w, h) = box.astype("int")

#                 # Calculate centroid of object
#                 cx = int(x + w/2)
#                 cy = int(y + h/2)
#                 centroid = (cx, cy)

#                 # Add centroid to dictionary
#                 curr_centroids[centroid] = True

#                 # Draw bounding box and centroid on frame
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                 cv2.circle(frame, centroid, 5, (0, 255, 0), -1)

#     # Calculate speed of each object based on centroids in current and previous frames
#     for centroid in curr_centroids:
#         if centroid in prev_centroids:
#             prev_centroid = prev_centroids[centroid]
#             distance = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))
#             speed = distance / 400.0 # Assuming 1 second between frames
#             print("Object speed:", speed)

#     # Store current centroids as previous centroids for next iteration
#     prev_frame = frame.copy()
#     prev_centroids = curr_centroids
    
#     # Calculate and display FPS
#     fps_frame_count += 1
#     if fps_start_time is None:
#         fps_start_time = time.time()
#     elif time.time() - fps_start_time >= 1:
#         fps = fps_frame_count / (time.time() - fps)
    
#     cv2.imshow("Video", frame)
#     cv2.waitKey(1)

import cv2
import numpy as np

# Load YOLOv4 weights and configuration
net = cv2.dnn.readNet("C:/Users/sonia/Desktop/speed/yolov4.weights", "C:/Users/sonia/Desktop/speed/yolov4.cfg")

# Load object class names
classes = []
with open("C:/Users/sonia/Desktop/speed/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set input size and scale factor
input_size = (608, 608)
scale_factor = 1/255.0

# Initialize variables for previous frame
prev_frame = None
prev_centroids = {}

# Start video capture
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('car_short.mp4')

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 220)

# Initialize timer for calculating FPS
timer = cv2.getTickCount()
fps = 0
fps_frame_count=0

while True:
    # Read frame from video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate FPS
    curr_time = cv2.getTickCount()
    exec_time = (curr_time - timer) / cv2.getTickFrequency()
    if exec_time != 0:
        fps = 1 / exec_time
    timer = curr_time

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
            speed = distance * fps / 5.2 # Assuming 1 second between frames
            print("Object speed:", speed)

    # Store current centroids as previous centroids for next iteration
    prev_frame = frame.copy()
    prev_centroids = curr_centroids
    
    
    cv2.imshow("Video", frame)
    cv2.waitKey(1)
