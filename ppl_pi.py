import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
from picamera import PiCamera
from picamera.array import PiRGBArray

model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

camera = PiCamera()
camera.resolution = (1020, 500)
raw_capture = PiRGBArray(camera, size=(1020, 500))

my_file = open("G:/yolov8peoplecounter-main/yolov8peoplecounter-main/coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

count = 0
tracker = Tracker()
total_people_count = 0  # New variable to store the total count

# File to store the total number of people counted
output_file = open("total_people_count.txt", "w")

for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
    image = frame.array
    count += 1

    if count % 3 == 0:
        results = model.predict(image)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")
        list = []

        for _, row in px.iterrows():
            x1, y1, x2, y2, d = map(int, row[:5])
            c = class_list[d]
            if 'person' in c:
                list.append([x1, y1, x2, y2])
            
        bbox_id = tracker.update(list)
        for bbox in bbox_id:
            x3, y3, x4, y4, _ = bbox
            cx = (x3 + x4) // 2
            cy = (y3 + y4) // 2
            cv2.circle(image, (cx, cy), 4, (255, 0, 255), -1)
        
        cv2.line(image, (3, 194), (1018, 194), (0, 255, 0), 2)
        cv2.line(image, (5, 220), (1019, 220), (0, 255, 255), 2)

        total_people_count += len(bbox_id)  # Update total count
        output_file.write(f"Total People Count: {total_people_count}\n")  # Write count to file

        cv2.imshow("RGB", image)

    raw_capture.truncate(0)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Close the output file
output_file.close()

cv2.destroyAllWindows()
