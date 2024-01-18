# Treee
#Lane detection using OpenCV (Image processing)

import os

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
# Grab the x and y size and make a copy of the image
# Read in the image
image = mpimg.imread('C:/Users/Administrator/PycharmProjects/robot/WhatsAppTest.jpg')

# Grab the x and y size and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
color_select = np.copy(image)


def lane_finding_pipeline(image):
    # Color selection
    red_threshold = 200
    green_threshold = 200
    blue_threshold = 200
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]
    color_select = np.copy(image)
    thresholds = (image[:, :, 0] < rgb_threshold[0]) | \
                 (image[:, :, 1] < rgb_threshold[1]) | \
                 (image[:, :, 2] < rgb_threshold[2])
    color_select[thresholds] = [0, 0, 0]

    # Region selection and drawing lines
    left_bottom = [0, 450]
    right_bottom = [1200, 550]
    apex = [450, 40]

    fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
    fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
    fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

    color_thresholds = (image[:, :, 0] < rgb_threshold[0]) | \
                       (image[:, :, 1] < rgb_threshold[1]) | \
                       (image[:, :, 2] < rgb_threshold[2])
    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    region_thresholds = (YY > (XX * fit_left[0] + fit_left[1])) & \
                        (YY > (XX * fit_right[0] + fit_right[1])) & \
                        (YY < (XX * fit_bottom[0] + fit_bottom[1]))
    color_select[color_thresholds | ~region_thresholds] = [0, 0, 0]

    # Plotting the triangular region
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.plot([left_bottom[0], right_bottom[0], apex[0], left_bottom[0]],
             [left_bottom[1], right_bottom[1], apex[1], left_bottom[1]], 'r--', lw=2)
    plt.title("Triangular Region")
    plt.show()

    # Edge detection and Hough transform
    gray = cv2.cvtColor(color_select, cv2.COLOR_RGB2GRAY)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    vertices = np.array([[(0, 450), (200, 100), (600, 100), (1200, 510)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)
    rho = 1
    theta = np.pi / 180
    threshold = 15
    min_line_length = 0
    max_line_gap = 200
    line_image = np.copy(image) * 0
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    # Check if lines is not None and contains at least one line
    if lines is not None and len(lines) > 0:
        # Iterate over the output "lines" and draw lines on a blank image
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10, cv2.LINE_AA)
    else:
        print("No lines detected.")

    color_edges = np.dstack((edges, edges, edges))
    lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)

    return image, color_select, lines_edges



video_path = 'C:/Users/Administrator/PycharmProjects/robot/WhatsApp5.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_frame, color_selected_frame, lane_lines_frame = lane_finding_pipeline(frame)

  
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB))
    plt.title("Input Frame")
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(color_selected_frame, cv2.COLOR_BGR2RGB))
    plt.title("Color Selection in the Triangular Region")
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(lane_lines_frame, cv2.COLOR_BGR2RGB))
    plt.title("Colored Lane line [In RED]")
    plt.show()

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
