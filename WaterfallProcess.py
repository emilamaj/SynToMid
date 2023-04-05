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
    cv2.imwrite("thresholded_image.png", filtered)
    return filtered

# This function returns the area inside the rectangle that is filled with white pixels, and the part that is filled with black pixels.
def match_area(image, rectangle):
    x, y, w, h = rectangle
    area = w * h
    white_area = np.sum(image[y:y + h, x:x + w] == 255)
    black_area = area - white_area
    return white_area, black_area

# This function tries to fit as little rectangles as possible inside the given rectangle, while maximizing the area coverage ratio.
def fit_rectangles(image, rectangle, max_rectangles=10):
    x, y, w, h = rectangle
    optimal_rectangles = [rectangle]
    white_area, black_area = match_area(image, rectangle)
    max_area_ratio = white_area / (white_area + black_area)

    for num_rectangles in range(2, max_rectangles + 1):
        best_area_ratio = 0
        best_combination = None

        for sizes in itertools.product(range(1, w), repeat=num_rectangles - 1):
            if sum(sizes) != w:
                continue

            candidate_rectangles = [(x, y, size, h) for size in sizes]

            sum_white_area = 0
            sum_black_area = 0
            for r in candidate_rectangles:
                w,b = match_area(image, r)
                sum_white_area += w
                sum_black_area += b
            candidate_ratio = sum_white_area / (sum_white_area + sum_black_area)

            if candidate_ratio > best_area_ratio:
                best_area_ratio = candidate_ratio
                best_combination = candidate_rectangles

        if best_area_ratio > max_area_ratio:
            max_area_ratio = best_area_ratio
            optimal_rectangles = best_combination
        else:
            break

    return optimal_rectangles

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
    _, sure_fg = cv2.threshold(dist_transform, 0.65 * dist_transform.max(), 255, 0)

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

    for marker in unique_markers:
        if marker == 0 or marker == 1:
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
                # print(f"Rectangle: x: {x}, y: {y}, w: {w}, h: {h}")
                # print(f"Area coverage ratio: {area_coverage_ratio}")

                if area_coverage_ratio < 0.05:
                    continue # Likely a rectangle enclosing the entire image.
                elif area_coverage_ratio < 0.98: # If the area has less than 98% white pixels, the rectangle probably encloses multiple keys.
                    sub_rectangles = fit_rectangles(image, rect)
                    rectangles.extend(sub_rectangles)
                else:
                    rectangles.append(rect)

    # return rectangles

    # To help debugging, write the rectangles on top of the image and save it to a file.
    for rect in rectangles:
        x, y, w, h = rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (150, 20, 200), 2) # If the color isn't visible, change the values.
    
    cv2.imwrite("rectangles.png", image)

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