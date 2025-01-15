import cv2
import numpy as np

# Video input and output
input_video_path = 'videos/input.mp4'
output_video_path = 'videos/output.mp4'

# Load the video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video

# Create VideoWriter to save the output video
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Color ranges in HSV
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])

lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Convert frame to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks
    mask_red = cv2.inRange(hsv_frame, lower_red, upper_red)
    mask_yellow = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
    mask_blue = cv2.inRange(hsv_frame, lower_blue, upper_blue)

    # Find contours
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Function to filter and annotate circular targets
    def process_contours(contours, color, label, frame):
        for contour in contours:
            # Calculate contour area and perimeter
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter > 0:  # Avoid division by zero
                circularity = 4 * np.pi * (area / (perimeter ** 2))
                if 0.7 < circularity <= 1.2:  # Filter by circularity
                    # Get the minimum enclosing circle
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))
                    radius = int(radius)
                    
                    if radius > 10:  # Ignore very small circles
                        # Draw the circle and label it
                        cv2.circle(frame, center, radius, color, 2)
                        cv2.putText(frame, label, (int(x - radius), int(y - radius - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Annotate red circles
    process_contours(contours_red, (0, 0, 255), "RED", frame)
    # Annotate yellow circles
    process_contours(contours_yellow, (0, 255, 255), "YELLOW", frame)
    # Annotate blue circles
    process_contours(contours_blue, (255, 0, 0), "BLUE", frame)

    # Write the frame to the output video
    out.write(frame)

    # Optional: Display the frame in a window (press 'q' to quit)
    cv2.imshow('Processed Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
