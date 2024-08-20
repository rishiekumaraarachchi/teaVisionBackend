import cv2
import numpy as np

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)
    circles = cv2.HoughCircles(
        binary,
        cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=50, param2=30, minRadius=0, maxRadius=0
    )
    mask = np.zeros_like(gray)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(mask, (x, y), r, 255, thickness=-1)
            break
    image_copy = np.full_like(image, (255, 255, 255), dtype=np.uint8)
    image_copy[mask == 255] = image[mask == 255]
    height, width, _ = image_copy.shape
    center_x, center_y = width // 2, height // 2
    margin_x = int(width * 0.15)
    margin_y = int(height * 0.15)
    start_x = max(center_x - margin_x, 0)
    end_x = min(center_x + margin_x, width)
    start_y = max(center_y - margin_y, 0)
    end_y = min(center_y + margin_y, height)
    cropped_image = image_copy[start_y:end_y, start_x:end_x]
    masked_pixels = image_copy[mask != 0]
    mean_color = np.mean(masked_pixels, axis=0)
    median_color = np.median(masked_pixels, axis=0)
    std_color = np.std(masked_pixels, axis=0)
    image_copy_hsb = cv2.cvtColor(image_copy.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    mean_hsb = np.mean(image_copy_hsb, axis=0)
    median_hsb = np.median(image_copy_hsb, axis=0)
    std_hsb = np.std(image_copy_hsb, axis=0)
    features = [mean_color.tolist()[0], mean_color.tolist()[1], mean_color.tolist()[2],
                median_color.tolist()[0], median_color.tolist()[1], median_color.tolist()[2],
                std_color.tolist()[0], std_color.tolist()[1], std_color.tolist()[2],
                mean_hsb.tolist()[0], mean_hsb.tolist()[1], mean_hsb.tolist()[2],
                median_hsb.tolist()[0], median_hsb.tolist()[1], median_hsb.tolist()[2],
                std_hsb.tolist()[0], std_hsb.tolist()[1], std_hsb.tolist()[2]]
    return features
