import numpy as np
import cv2
import matplotlib.pyplot as plt


def main():
    image = cv2.imread('photos/3.jpg')

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0,100,100])
    upper_red = np.array([10, 100, 100])

    lower_yellow = np.array([173, 151, 57])
    upper_yellow = np.array([251, 255, 0])

    lower_blue = np.array([220, 100, 100])
    upper_blue = np.array([250, 100, 100]) 


    mask_red = cv2.inRange(hsv_image, lower_red, upper_red)
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)


    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours_red:
        x, y, w, h = cv2.boundingRect(contour)
        label = "RED"
        text_position = (x, y - 10)
        cv2.putText(image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # Red color
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    for contour in contours_yellow:
        x, y, w, h = cv2.boundingRect(contour)
        label = "YELLOW"
        text_position = (x, y - 10)
        cv2.putText(image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)  # Yellow color
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2) 
    for contour in contours_blue:
        x, y, w, h = cv2.boundingRect(contour)
        label = "BLUE"
        text_position = (x, y - 10)
        cv2.putText(image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)  # Blue color
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2) 
    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("test")
    plt.axis('off')
    plt.show()



if __name__ == "__main__":
    main()