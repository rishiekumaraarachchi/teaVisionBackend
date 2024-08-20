import cv2
import numpy as np

def is_contour_inner(contour_index, hierarchy):
    return hierarchy[0][contour_index][3] != -1

def identify_stroke_in_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 300

    # Define RGB ranges for brown particles
    r_lower_bound, r_upper_bound = 80, 200
    g_lower_bound, g_upper_bound = 40, 150
    b_lower_bound, b_upper_bound = 20, 150

    filtered_contours = []
    all_r_values = []
    all_g_values = []
    all_b_values = []
    brown_r_values = []
    brown_g_values = []
    brown_b_values = []

    contour_areas = [cv2.contourArea(contour) for contour in contours if cv2.contourArea(contour) > 0]
    percentile_90_area = np.percentile(contour_areas, 90)
    large_contour_area = percentile_90_area * 1.5

    for i, contour in enumerate(contours):
        contour_area = cv2.contourArea(contour)
        if min_contour_area < contour_area < large_contour_area and not is_contour_inner(i, hierarchy):
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)

            r_values = image[mask == 255][:, 2]
            g_values = image[mask == 255][:, 1]
            b_values = image[mask == 255][:, 0]

            mean_r = np.mean(r_values)
            mean_g = np.mean(g_values)
            mean_b = np.mean(b_values)

            all_r_values.extend(r_values)
            all_g_values.extend(g_values)
            all_b_values.extend(b_values)

            if (r_lower_bound <= mean_r <= r_upper_bound and
                g_lower_bound <= mean_g <= g_upper_bound and
                b_lower_bound <= mean_b <= b_upper_bound):
                filtered_contours.append(contour)
                brown_r_values.extend(r_values)
                brown_g_values.extend(g_values)
                brown_b_values.extend(b_values)

    filtered_background = np.full_like(image, (255, 255, 255), dtype=np.uint8)
    for contour in filtered_contours:
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        filtered_background[mask == 255] = image[mask == 255]

    stats = {
        "number_of_external_contours": sum(
            1 for i in range(len(contours))
            if min_contour_area < cv2.contourArea(contours[i]) < large_contour_area and not is_contour_inner(i, hierarchy)
        ),
        "number_of_brown_particles": len(filtered_contours),
        "brown_particle_ratio": len(filtered_contours) / (
            sum(
                1 for i in range(len(contours))
                if min_contour_area < cv2.contourArea(contours[i]) < large_contour_area and not is_contour_inner(i, hierarchy)
            ) if sum(
                1 for i in range(len(contours))
                if min_contour_area < cv2.contourArea(contours[i]) < large_contour_area and not is_contour_inner(i, hierarchy)
            ) > 0 else 0
        ) * 100
    }

    return filtered_background, stats
