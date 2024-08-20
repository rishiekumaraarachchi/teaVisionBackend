import cv2
import numpy as np



def calculate_longest_distance(contour):
    max_distance = 0
    for i in range(len(contour)):
        for j in range(i + 1, len(contour)):
            distance = np.linalg.norm(contour[i][0] - contour[j][0])
            if distance > max_distance:
                max_distance = distance
    return max_distance

def is_contour_touching_boundary(contour, image_shape):
    for point in contour:
        if point[0][0] <= 0 or point[0][0] >= image_shape[1] - 1 or point[0][1] <= 0 or point[0][1] >= image_shape[0] - 1:
            return True
    return False

def identify_fiber_in_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 300
    max_aspect_ratio = 11

    segmented_image = np.full_like(image, (255, 255, 255), dtype=np.uint8)
    mask = np.zeros_like(gray)
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > min_contour_area:
            cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    segmented_image[mask == 255] = image[mask == 255]

    thin_particles = []
    thin_particles_image = np.full_like(image, (255, 255, 255), dtype=np.uint8)

    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] == -1 and not is_contour_touching_boundary(contour, image.shape):
            area = cv2.contourArea(contour)
            if area > min_contour_area:
                height = calculate_longest_distance(contour)
                aspect_ratio = area / height

                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = contour[0][0][0], contour[0][0][1]

                text = f"A: {int(area)} H: {int(height)}"
                cv2.putText(segmented_image, text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                if aspect_ratio < max_aspect_ratio:
                    thin_particles.append(contour)
                    mask = np.zeros_like(gray, dtype=np.uint8)
                    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
                    thin_particles_image[mask == 255] = image[mask == 255]
                    cv2.putText(thin_particles_image, text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Return the result image and statistics
    stats = {
        "number_of_thin_particles": len(thin_particles),
        "total_number_of_particles": sum(
            1 for i in range(len(contours))
            if cv2.contourArea(contours[i]) > min_contour_area and not is_contour_touching_boundary(contours[i], image.shape) and hierarchy[0][i][3] == -1
        ),
        "average_ratio_thin_to_total": len(thin_particles) / sum(
            1 for i in range(len(contours))
            if cv2.contourArea(contours[i]) > min_contour_area and not is_contour_touching_boundary(contours[i], image.shape) and hierarchy[0][i][3] == -1
        ) * 100 if sum(
            1 for i in range(len(contours))
            if cv2.contourArea(contours[i]) > min_contour_area and not is_contour_touching_boundary(contours[i], image.shape) and hierarchy[0][i][3] == -1
        ) > 0 else 0
    }

    return thin_particles_image, stats
