import cv2
import numpy as np
import sys
import argparse
import mido
from mido import Message, MidiFile, MidiTrack
import itertools

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
def match_area(image, rectangle):
    x, y, w, h = rectangle
    area = w * h
    white_area = np.sum(image[y:y + h, x:x + w] == 255)
    black_area = area - white_area
    return white_area, black_area

# This function tries to fit as little rectangles as possible inside the given rectangle, while maximizing the area coverage ratio.

def fit_rectangles(image, rectangle, min_cover_ratio=0.95, max_rectangles=10):
    x, y, w, h = rectangle
    shape_roi = image[y:y+h, x:x+w]

    contours, _ = cv2.findContours(shape_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []

    largest_contour = max(contours, key=cv2.contourArea)
    shape_mask = np.zeros_like(shape_roi)
    cv2.drawContours(shape_mask, [largest_contour], -1, 255, -1)

    fitted_rectangles = []
    current_cover = 0
    initial_area = cv2.contourArea(largest_contour)
    target_area = initial_area * min_cover_ratio

    while current_cover < target_area and len(fitted_rectangles) < max_rectangles:
        min_area_rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(min_area_rect)
        box = np.int0(box)

        fitted_rect = cv2.boundingRect(box)
        fitted_rectangles.append(fitted_rect)

        rx, ry, rw, rh = fitted_rect
        cv2.rectangle(shape_mask, (rx, ry), (rx + rw, ry + rh), 0, -1)

        contours, _ = cv2.findContours(shape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            break

        largest_contour = max(contours, key=cv2.contourArea)
        current_cover = initial_area - cv2.contourArea(largest_contour)

    # Offset the fitted rectangles to the original image coordinates
    fitted_rectangles = [(x + rx, y + ry, rw, rh) for rx, ry, rw, rh in fitted_rectangles]
    
    return fitted_rectangles

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
        if marker == 0 or marker == 1:
            continue

        mask = np.zeros_like(image, dtype=np.uint8)
        mask[markers == marker] = 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 0 and h > 0:
                
                # # HACK: Remove
                # if len(rectangles) > 4:
                #     break

                rect = (x, y, w, h)
                white_area, black_area = match_area(image, rect)
                area_coverage_ratio = white_area / (white_area + black_area)
                print(f"Rectangle: x: {x}, y: {y}, w: {w}, h: {h}")
                print(f"Area coverage ratio: {area_coverage_ratio}")

                if area_coverage_ratio < 0.05:
                    print(f"### Skipping rectangle: {rect}")
                    continue # Likely a rectangle enclosing the entire image.
                elif area_coverage_ratio < min_cover_ratio: # If the area has less than 98% white pixels, the rectangle probably encloses multiple keys.
                    sub_rectangles = fit_rectangles(image, rect, min_cover_ratio, 10)
                    rectangles = rectangles + sub_rectangles
                    print(f"Subrectangles: {sub_rectangles}")
                else:
                    rectangles.append(rect)
                    print("Single rectangle.")


    # return rectangles

    # To help debugging, write the rectangles on top of the image and save it to a file.
    for rect in rectangles:
        x, y, w, h = rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (150, 20, 200), 2) # If the color isn't visible, change the values.
    
    cv2.imwrite("debug_rectangles.png", image)

    return rectangles

def convert_notes(rectangles, width, duration, transpose):
    """
    Convert the detected rectangles to notes.
    """
    notes = []

    for rectangle in rectangles:
        x, y, w, h = rectangle
        key = round(x / width * 88) + 21 + transpose
        note_start = round(y / width * duration)
        note_end = round((y + h) / width * duration)
        notes.append((key, note_start, note_end))

    return notes

def write_midi(notes, filename):
    """
    Write the notes to a MIDI file.
    """
    midi_file = MidiFile()
    track = MidiTrack()
    midi_file.tracks.append(track)

    for note in notes:
        key, note_start, note_end = note
        track.append(Message("note_on", note=key, velocity=64, time=note_start))
        track.append(Message("note_off", note=key, velocity=64, time=note_end))

    midi_file.save(filename)

def main():
    parser = argparse.ArgumentParser(description="Convert a Synthesia-style piano roll image to a MIDI file.")
    parser.add_argument("image_file", help="Input image file")
    parser.add_argument("output_file", help="Output MIDI file")
    parser.add_argument("duration", type=int, help="Image duration in seconds")
    parser.add_argument("transpose", type=int, help="Transpose value in semitones (can be negative)")

    args = parser.parse_args()

    image = preprocess_image(args.image_file)
    rectangles = detect_rectangles(image)
    notes = convert_notes(rectangles, image.shape[1], args.duration, args.transpose)
    write_midi(notes, args.output_file)

if __name__ == "__main__":
    main()

# To run the script, use the following command:
# python WaterfallProcess.py input.png output.mid 30 0