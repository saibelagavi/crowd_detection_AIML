import cv2
import numpy as np

# Load the pre-trained YOLO model for object detection
net = cv2.dnn.readNet("G:/yolov8peoplecounter-main/yolov8peoplecounter-main/yolov3.weights", "G:/yolov8peoplecounter-main/yolov8peoplecounter-main/yolov3.cfg")
classes = []
with open("G:/yolov8peoplecounter-main/yolov8peoplecounter-main/coco.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getUnconnectedOutLayersNames()

# Initialize the video capture
cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera, change if using a different camera

# Initialize variables to store head count
previous_head_count = 0
current_head_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Get the height and width of the frame
    height, width, _ = frame.shape

    # Convert the frame to a blob for YOLO input
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    # Initialize lists for detected heads
    class_ids = []
    confidences = []
    boxes = []

    # Loop over each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Class ID 0 corresponds to the 'person' class
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Get the coordinates of the top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to eliminate overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Count the number of heads
    current_head_count = len(indexes)

    # Draw rectangles around the detected heads
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the frame with rectangles and head count
    cv2.putText(frame, f'Head Count: {current_head_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)

    # Check for an increase in head count
    if current_head_count > previous_head_count:
        # Update the previous head count
        previous_head_count = current_head_count

        # Store the total head count in a file
        try:
            with open('head_count.txt', 'w') as file:
                file.write(str(current_head_count))
            print("Head count successfully saved to 'head_count.txt'.")
        except Exception as e:
            print(f"Error saving head count to file: {e}")

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
