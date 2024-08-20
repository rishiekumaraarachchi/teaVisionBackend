import numpy as np
import cv2
from PIL import Image
import joblib
import os

# Load the RandomForest model and label encoder
model_path = 'models/'
size_rf = joblib.load(os.path.join(model_path, 'rf_size.pkl'))
size_label_encoder = joblib.load(os.path.join(model_path, 'size_label_encoder.pkl'))

def is_contour_touching_boundary(contour, image_shape):
    for point in contour:
        if point[0][0] <= 0 or point[0][0] >= image_shape[1] - 1 or point[0][1] <= 0 or point[0][1] >= image_shape[0] - 1:
            return True
    return False

def get_contour_dimensions(contour):
    x, y, w, h = cv2.boundingRect(contour)
    if w > h:
        return w, h  # width is greater, so width becomes height
    else:
        return h, w  # height is greater, so height remains height

def process_image(image):
    # Convert image to OpenCV format
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Thresholding
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    # Find contours and hierarchy
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # Calculate the area of each contour
    contour_areas = [cv2.contourArea(contour) for contour in contours if cv2.contourArea(contour) > 0]
    # Determine the 90th percentile of the contour areas
    percentile_90_area = np.percentile(contour_areas, 90)
    large_contour_area = percentile_90_area * 1.5  # Set large_contour_area to 1.5 times the 90th percentile area
    # Define a minimum contour area to filter out small particles
    min_contour_area = 300  # Adjust this threshold as needed
    small_particle_areas = []
    widths = []
    heights = []
    # Create a white background image
    white_background = np.full_like(image, (255, 255, 255), dtype=np.uint8)
    # Copy the original image to the white background where the mask is
    mask = np.zeros_like(gray)
    # Process external contours for small particles
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > min_contour_area:
            cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    white_background[mask == 255] = image[mask == 255]
    # Draw contours, bounding boxes, and label the particles
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if min_contour_area < area < large_contour_area and hierarchy[0][i][3] == -1 and not is_contour_touching_boundary(contour, image.shape):
            small_particle_areas.append(area)
            height, width = get_contour_dimensions(contour)
            widths.append(width)
            heights.append(height)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(white_background, f"Area: {int(area)}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(white_background, f"Width: {int(width)}", (cX, cY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(white_background, f"Height: {int(height)}", (cX, cY + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.drawContours(white_background, [contour], -1, (0, 0, 255), 2)  # Draw small particle contours in red
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(white_background, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw bounding box in green
    return small_particle_areas, widths, heights

def predict_tea_variant(image):
    small_particle_areas, widths, heights = process_image(image)

    all_small_particle_areas = []
    all_widths = []
    all_heights = []

    all_small_particle_areas.extend(small_particle_areas)
    all_widths.extend(widths)
    all_heights.extend(heights)

    if all_small_particle_areas:
        mean_area_small_particles_all_images = np.mean(all_small_particle_areas)
        median_area_small_particles_all_images = np.median(all_small_particle_areas)
        std_dev_area_small_particles_all_images = np.std(all_small_particle_areas)
    else:
        mean_area_small_particles_all_images = 0
        median_area_small_particles_all_images = 0
        std_dev_area_small_particles_all_images = 0

    if all_widths:
        mean_width_all_images = np.mean(all_widths)
        median_width_all_images = np.median(all_widths)
        std_dev_width_all_images = np.std(all_widths)
    else:
        mean_width_all_images = 0
        median_width_all_images = 0
        std_dev_width_all_images = 0

    if all_heights:
        mean_height_all_images = np.mean(all_heights)
        median_height_all_images = np.median(all_heights)
        std_dev_height_all_images = np.std(all_heights)
    else:
        mean_height_all_images = 0
        median_height_all_images = 0
        std_dev_height_all_images = 0

    features = [
        len(small_particle_areas),
        sum(small_particle_areas),
        np.mean(small_particle_areas) if small_particle_areas else 0,
        np.median(small_particle_areas) if small_particle_areas else 0,
        np.std(small_particle_areas) if small_particle_areas else 0,
        np.mean(widths) if widths else 0,
        np.median(widths) if widths else 0,
        np.std(widths) if widths else 0,
        np.mean(heights) if heights else 0,
        np.median(heights) if heights else 0,
        np.std(heights) if heights else 0,
        mean_area_small_particles_all_images,
        median_area_small_particles_all_images,
        std_dev_area_small_particles_all_images,
        mean_width_all_images,
        median_width_all_images,
        std_dev_width_all_images,
        mean_height_all_images,
        median_height_all_images,
        std_dev_height_all_images
    ]

    y_pred_rf = size_rf.predict([features])
    t_variant = size_label_encoder.inverse_transform([y_pred_rf[0]])

    return t_variant[0]
