import cv2
import numpy as np
import sys
import argparse
import json


def preprocess_image(image_file):
    """
    Preprocess the input image to remove noise and threshold the keys.
    Returns a numpy array of the preprocessed image and saves the result to a file.
    """
    image = cv2.imread(image_file, 0)
    _, thresholded = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    filtered = cv2.medianBlur(thresholded, 5)
    cv2.imwrite("debug_threshold.png", filtered)
    return filtered

# This function returns the area inside the rectangle that is filled with white pixels, and the part that is filled with black pixels.
def match_area(image, rectangle, outer_rectangle=None):
    # If rectangle is a list of rectangles, return the sum of the areas of all rectangles. (Intersections are not counted twice.)
    if isinstance(rectangle, list) and (isinstance(rectangle[0], list) or isinstance(rectangle[0], tuple)):
        # Find the bounding box of all rectangles.
        x, y, w, h = outer_rectangle
        area = w * h

        # Create a mask of the rectangles
        intersection_mask = np.zeros(image.shape, dtype=np.uint8)
        intersection_mask = intersection_mask[y:y + h, x:x + w]

        for rect in rectangle:
            # Fill the mask with white pixels where rect is located.
            x1, y1, w1, h1 = rect
            intersection_mask[y1 - y:y1 - y + h1, x1 - x:x1 - x + w1] = 255

        # Count the number of white pixels that are both in the mask and in the image.
        white_area = np.sum(np.logical_and(image[y:y + h, x:x + w] == 255, intersection_mask == 255))
        black_area = area - white_area

        return white_area, black_area
    else:
        x, y, w, h = rectangle
        area = w * h
        white_area = np.sum(image[y:y + h, x:x + w] == 255)
        black_area = area - white_area
        return white_area, black_area

def separate_contours(image, contour, threshold=10):
    """
    Separates merged contours into individual convex contours.

    Args:
    image (np.array): Binary image containing the contours.
    contour (list): A list of points representing the contour to be separated.
    threshold (float): Distance threshold for convexity defect points to be considered as separation points.

    Returns:
    list: A list of individual convex contours.
    """

    # Find convex hull
    hull = cv2.convexHull(contour, returnPoints=False)

    # Calculate convexity defects
    defects = cv2.convexityDefects(contour, hull)

    if defects is None:
        return [contour]

    separation_points = []

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        if d > threshold:
            separation_points.append(f)

    # If no separation points are found, return the original contour
    if len(separation_points) == 0:
        return [contour]

    separated_contours = []
    separation_points.append(separation_points[0])  # Close the loop

    # Create separate contours based on the separation points
    for i in range(len(separation_points) - 1):
        start = separation_points[i]
        end = separation_points[i + 1]
        separated_contour = contour[start:end]
        separated_contours.append(separated_contour)

    return separated_contours

def subdivide_rectangle(image, contour):
    """
    Takes a contour that mistakook multiple keys for a single contour, and subdivides it into the correct note rectangle contours.
    """
    bounding_rect = cv2.boundingRect(contour)

    # Use function separate_contours with different threshold values, and keep the result with the best area ratio.
    best_ratio = 0
    best_separated_contours = None

    for thres in range(1, 100, 1):
        separated_contours = separate_contours(image, contour, threshold=thres)

        # Calculate the area ratio of the separated contours
        rectangle_list = [cv2.boundingRect(c) for c in separated_contours]

        # Max white area
        max_white_area, black_area = match_area(image, bounding_rect)
        aw, ab = match_area(image, rectangle_list, bounding_rect)
        area_ratio = aw / max_white_area

        if area_ratio > best_ratio:
            best_ratio = area_ratio
            best_separated_contours = separated_contours

    # Convert the contours to rectangles
    rectangle_list = [cv2.boundingRect(c) for c in best_separated_contours]

    # Keep only the rectangles that are within the bounding rectangle
    rectangle_list = [r for r in rectangle_list if r[0] >= bounding_rect[0] and r[1] >= bounding_rect[1] and r[0] + r[2] <= bounding_rect[0] + bounding_rect[2] and r[1] + r[3] <= bounding_rect[1] + bounding_rect[3]]

    # Keep only the rectangles that are not too thin
    rectangle_list = [r for r in rectangle_list if r[2] >= 5]

    print(f"Area ratio: {best_ratio}")

    return rectangle_list


def detect_rectangles(image):
    """
    Detect the rectangles corresponding to the keys using Watershed Algorithm.
    Handles cases where keys may be touching.
    Returns a list of rectangles.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(image, kernel, iterations=0)

    # Distance transform and normalization
    dist_transform = cv2.distanceTransform(dilated, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.35 * dist_transform.max(), 255, 0)

    # Finding sure background area
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(dilated, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Apply the watershed
    cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers)

    # Get the list of unique markers
    unique_markers = np.unique(markers)

    rectangles = []
    min_cover_ratio = 0.95

    for marker in unique_markers:
        if marker == 0 or marker == 1: # Skip the background and the foreground.
            continue

        mask = np.zeros_like(image, dtype=np.uint8)
        mask[markers == marker] = 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 0 and h > 0:

                rect = (x, y, w, h)
                white_area, black_area = match_area(image, rect)
                area_coverage_ratio = white_area / (white_area + black_area)
                print(f"Rectangle: x: {x}, y: {y}, w: {w}, h: {h}")
                print(f"Area coverage ratio: {area_coverage_ratio}")

                if area_coverage_ratio < 0.05:
                    print(f"### Skipping rectangle: {rect}")
                    continue # Likely a rectangle enclosing the entire image.
                elif area_coverage_ratio < min_cover_ratio:
                    print(f"### Subdividing rectangle: {rect}")
                    # The rectangle may be covering multiple keys.
                    # Subdivide the rectangle into multiple rectangles.
                    rects = subdivide_rectangle(image, contour)
                    print(f"### Result: {rects}")
                    rectangles += rects
                else:
                    rectangles.append(rect)
                    print("Single rectangle.")

    return rectangles

def write_image(image, rectangles, output_file):
    """
    To help debugging, write the rectangles on top of the image and save it to a file.
    """
    for rect in rectangles:
        x, y, w, h = rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (150, 150, 150), 2) # If the color isn't visible, change the values.
    
    cv2.imwrite(output_file, image)

    return rectangles

def write_json(rectangles, output_file):
    """
    Write the rectangles to a json file.
    """

    rects = []
    for rect in rectangles:
        x, y, w, h = rect
        rects.append({"x": x, "y": y, "width": w, "height": h})
    data = {"rectangles": rects}

    with open(output_file, "w") as f:
        json.dump(data, f)

def main():
    parser = argparse.ArgumentParser(description="Convert a Synthesia-style piano roll image to a list of rectangles representing the notes.")
    parser.add_argument("image_file", help="Input image file")
    parser.add_argument("output_process", help="Debug image file showing the detected rectangles")
    parser.add_argument("output_file", help="Output json file listing the rectangles")

    args = parser.parse_args()

    image = preprocess_image(args.image_file)
    rectangles = detect_rectangles(image)
    write_image(image, rectangles, args.output_process)
    write_json(rectangles, args.output_file)

if __name__ == "__main__":
    main()

# To run the script, use the following command:
# python WaterfallProcess.py output_stitch.png output_process.png rectangles.json