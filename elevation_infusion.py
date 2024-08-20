import cv2
import numpy as np

def color_features_infusion_predict(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 300
    white_background = np.full_like(image, (255, 255, 255), dtype=np.uint8)
    mask = np.zeros_like(gray)
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > min_contour_area:
            cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    white_background[mask == 255] = image[mask == 255]
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > min_contour_area:
            if hierarchy[0][i][3] != -1:
                cv2.drawContours(white_background, [contour], -1, (255, 255, 255), cv2.FILLED)
    masked_pixels = white_background[mask != 0]
    non_white_pixels = masked_pixels[(masked_pixels[:, 0] < 255) | (masked_pixels[:, 1] < 255) | (masked_pixels[:, 2] < 255)]
    mean_color = np.mean(non_white_pixels, axis=0)
    median_color = np.median(non_white_pixels, axis=0)
    std_color = np.std(non_white_pixels, axis=0)
    non_white_pixels_hsb = cv2.cvtColor(non_white_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    mean_hsb = np.mean(non_white_pixels_hsb, axis=0)
    median_hsb = np.median(non_white_pixels_hsb, axis=0)
    std_hsb = np.std(non_white_pixels_hsb, axis=0)
    features = [mean_color.tolist()[0], mean_color.tolist()[1], mean_color.tolist()[2],
                median_color.tolist()[0], median_color.tolist()[1], median_color.tolist()[2],
                std_color.tolist()[0], std_color.tolist()[1], std_color.tolist()[2],
                mean_hsb.tolist()[0], mean_hsb.tolist()[1], mean_hsb.tolist()[2],
                median_hsb.tolist()[0], median_hsb.tolist()[1], median_hsb.tolist()[2],
                std_hsb.tolist()[0], std_hsb.tolist()[1], std_hsb.tolist()[2]]
    return features
