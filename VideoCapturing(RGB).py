import cv2
import numpy as np

# Define color ranges in HSV for object detection
color_ranges = {
    "red": [(0, 50, 50), (10, 255, 255)],  # Lower and upper range for bright red
    "red_2": [(170, 50, 50), (180, 255, 255)],  # Additional range for dark red
    "blue": [(100, 150, 0), (140, 255, 255)],  # Lower and upper range for blue
    "yellow": [(20, 100, 100), (30, 255, 255)],  # Lower and upper range for yellow
}

# Input video file
input_video = "/home/jscy/Downloads/Video1.mp4"  # Replace with your video filename
output_video = "output_video_with_detection.mp4"

# Video capture and writer
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# Background subtractor for motion detection
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction to ignore static objects
    foreground_mask = background_subtractor.apply(frame)
    foreground_mask = cv2.threshold(foreground_mask, 127, 255, cv2.THRESH_BINARY)[1]  # Binarize the mask
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # Clean noise

    # Convert the frame to HSV for color detection
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect and highlight targets
    detected_targets = []
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
        if color == "red_2":  # Combine red ranges
            mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper)) | mask

        # Apply the foreground mask to isolate moving objects
        mask = cv2.bitwise_and(mask, mask, mask=foreground_mask)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Minimum area to consider a detection
                x, y, w, h = cv2.boundingRect(contour)
                detected_targets.append((x, y, w, h, color))

    # Highlight detected targets
    for x, y, w, h, color in detected_targets:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box
        cv2.putText(
            frame,
            f"{color.split('_')[0]} target",  # Label (ignore "_2" in red)
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Show the frame in real-time
    cv2.imshow("Detection", frame)

    # Write the frame to the output video
    out.write(frame)

    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video with detections has been saved as", output_video)
