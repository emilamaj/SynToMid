import cv2
import numpy as np
import sys
import argparse
import mido
from mido import Message, MidiFile, MidiTrack

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

# This function tries to fit multiple sub-rectangles inside the given rectangle, such that the area covered by the rectangles is at least min_cover_ratio.
def fit_rectangles(image, rectangle, min_cover_ratio=0.95, max_rectangles=3):
    # Divide outer_rectangle vertically into num_rectangles parts, of the same width and height
    def initial_guess(num_rectangles, outer_rectangle):
        x, y, w, h = outer_rectangle
        rectangles = []
        for i in range(num_rectangles):
            rectangles.append([x, y + i * h // num_rectangles, w, h // num_rectangles])
        return rectangles

    # Move individual rectangles in the position that maximize the area covered by white pixels.
    def optimize_position(rectangles, image, outer_rectangle, amplitude=1):
        combined_area_white, combined_area_black = match_area(image, rectangles, outer_rectangle=rectangle)
        combined_cover_ratio = combined_area_white / (combined_area_white + combined_area_black)

        for i in range(len(rectangles)):
            # Process rectangle i
            x, y, w, h = rectangles[i]
            position_ratios = [] # Stores the area covered by white pixels for each position, to select the best position.
            temprect = rectangles.copy()

            # Move the rectangle to the left.
            # Check if the rectangle is still inside the outer rectangle.
            if x - amplitude >= outer_rectangle[0]:
                temprect = rectangles.copy()
                temprect[i] = (x - amplitude, y, w, h)
                larea_white, larea_black = match_area(image, temprect, outer_rectangle=rectangle)
                # Avoid division by zero.
                if larea_white + larea_black == 0:
                    position_ratios.append(0)
                else:
                    position_ratios.append(larea_white / (larea_white + larea_black))
            else:
                position_ratios.append(0)

            # Move the rectangle to the right.
            # Check if the rectangle is still inside the outer rectangle.
            if x + w + amplitude <= outer_rectangle[0] + outer_rectangle[2]:
                temprect = rectangles.copy()
                temprect[i] = (x + amplitude, y, w, h)
                rarea_white, rarea_black = match_area(image, temprect, outer_rectangle=rectangle)
                # Avoid division by zero.
                if rarea_white + rarea_black == 0:
                    position_ratios.append(0)
                else:
                    position_ratios.append(rarea_white / (rarea_white + rarea_black))
            else:
                position_ratios.append(0)

            # Move the rectangle up.
            # Check if the rectangle is still inside the outer rectangle.
            if y - amplitude >= outer_rectangle[1]:
                temprect = rectangles.copy()
                temprect[i] = (x, y - amplitude, w, h)
                uarea_white, uarea_black = match_area(image, temprect, outer_rectangle=rectangle)
                # Avoid division by zero.
                if uarea_white + uarea_black == 0:
                    position_ratios.append(0)
                else:
                    position_ratios.append(uarea_white / (uarea_white + uarea_black))
            else:
                position_ratios.append(0)

            # Move the rectangle down.
            # Check if the rectangle is still inside the outer rectangle.
            if y + h + amplitude <= outer_rectangle[1] + outer_rectangle[3]:
                temprect = rectangles.copy()
                temprect[i] = (x, y + amplitude, w, h)
                darea_white, darea_black = match_area(image, temprect, outer_rectangle=rectangle)
                # Avoid division by zero.
                if darea_white + darea_black == 0:
                    position_ratios.append(0)
                else:
                    position_ratios.append(darea_white / (darea_white + darea_black))
            else:
                position_ratios.append(0)

            # Find the best position for the rectangle.
            best_position = np.argmax(position_ratios)
            # Compare the best position with the current position.
            if combined_cover_ratio > position_ratios[best_position]:
                return 

            # Move the rectangle to the best position.
            if best_position == 0:
                rectangles[i] = [x - amplitude, y, w, h]
            elif best_position == 1:
                rectangles[i] = [x + amplitude, y, w, h]
            elif best_position == 2:
                rectangles[i] = [x, y - amplitude, w, h]
            elif best_position == 3:
                rectangles[i] = [x, y + amplitude, w, h]

    # Change the size of individual rectangles to maximize the area covered by white pixels.
    def optimize_shape(rectangles, image, outer_rectangle, amplitude=1):
        combined_area_white, combined_area_black = match_area(image, rectangles, outer_rectangle=outer_rectangle)
        combined_cover_ratio = combined_area_white / (combined_area_white + combined_area_black)

        for i in range(len(rectangles)):
            # Process rectangle i
            x, y, w, h = rectangles[i]
            shape_ratios = [] # Stores the area covered by white pixels for each shape, to select the best shape.
            

            # Increase the width of the rectangle.
            # Check if the rectangle is still inside the outer rectangle.
            if x + w + amplitude <= outer_rectangle[0] + outer_rectangle[2]:
                temprect = rectangles.copy()
                temprect[i] = (x, y, w + amplitude, h)
                areas_white, areas_black = match_area(image, temprect, outer_rectangle=outer_rectangle)
                # Avoid division by zero.
                if areas_white + areas_black == 0:
                    shape_ratios.append(0)
                else:
                    shape_ratios.append(areas_white / (areas_white + areas_black))
            else:
                shape_ratios.append(0)

            # Decrease the width of the rectangle.
            # Check if the rectangle still has a positive width.
            if w - amplitude > 0:
                temprect = rectangles.copy()
                temprect[i] = (x, y, w - amplitude, h)
                larea_white, larea_black = match_area(image, temprect, outer_rectangle=outer_rectangle)
                # Avoid division by zero.
                if larea_white + larea_black == 0:
                    shape_ratios.append(0)
                else:
                    shape_ratios.append(larea_white / (larea_white + larea_black))
            else:
                shape_ratios.append(0)

            # Increase the height of the rectangle.
            # Check if the rectangle is still inside the outer rectangle.
            if y + h + amplitude <= outer_rectangle[1] + outer_rectangle[3]:
                temprect = rectangles.copy()
                temprect[i] = (x, y, w, h + amplitude)
                darea_white, darea_black = match_area(image, temprect, outer_rectangle=outer_rectangle)
                # Avoid division by zero.
                if darea_white + darea_black == 0:
                    shape_ratios.append(0)
                else:
                    shape_ratios.append(darea_white / (darea_white + darea_black))
            else:
                shape_ratios.append(0)

            # Decrease the height of the rectangle.
            # Check if the rectangle still has a positive height.
            if h - amplitude > 0:
                temprect = rectangles.copy()
                temprect[i] = (x, y, w, h - amplitude)
                uarea_white, uarea_black = match_area(image, temprect, outer_rectangle=outer_rectangle)
                # Avoid division by zero.
                if uarea_white + uarea_black == 0:
                    shape_ratios.append(0)
                else:
                    shape_ratios.append(uarea_white / (uarea_white + uarea_black))
            else:
                shape_ratios.append(0)

            # Find the best shape for the rectangle.
            best_shape = np.argmax(shape_ratios)

            # Compare the best shape with the current shape.
            if combined_cover_ratio > shape_ratios[best_shape]:
                return

            # Change the shape of the rectangle to the best shape.
            if best_shape == 0:
                rectangles[i] = [x, y, w + amplitude, h]
            elif best_shape == 1:
                rectangles[i] = [x, y, w - amplitude, h]
            elif best_shape == 2:
                rectangles[i] = [x, y, w, h + amplitude]
            elif best_shape == 3:
                rectangles[i] = [x, y, w, h - amplitude]

    # Calculate max area to be covered by white pixels.
    ref_white, ref_black = match_area(image, rectangle)

    for num_rectangles in range(1, max_rectangles + 1):
        rectangles = initial_guess(num_rectangles, rectangle)
        for step in range(100):  # Repeat optimization steps 10 times
            # if step < 20:
            #     ampl = 10
            # elif step < 40:
            #     ampl = 5
            # elif step < 70:
            #     ampl = 3
            # else:
            ampl = 1
            
            optimize_position(rectangles, image, rectangle, amplitude=ampl)
            optimize_shape(rectangles, image, rectangle, amplitude=ampl)
            areas_white, areas_black = match_area(image, rectangles, outer_rectangle=rectangle)
            cover_ratio = areas_white / (areas_white + areas_black)
            print(rectangles)
            print(f"Areas white: {areas_white}, Areas black: {areas_black}, Cover ratio: {cover_ratio}")

            if cover_ratio >= min_cover_ratio:
                print(f"Found {num_rectangles} rectangles. with cover ratio {cover_ratio}")
                return rectangles
        
        print(f"Best cover ratio {cover_ratio} found with {num_rectangles} rectangles.")

#HACK: REMOVE THIS
    # return [rectangle]
    return rectangles

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
                
                # HACK: Remove
                if len(rectangles) > 4:
                    break

                rect = (x, y, w, h)
                white_area, black_area = match_area(image, rect)
                area_coverage_ratio = white_area / (white_area + black_area)
                print(f"Rectangle: x: {x}, y: {y}, w: {w}, h: {h}")
                print(f"Area coverage ratio: {area_coverage_ratio}")

                if area_coverage_ratio < 0.05:
                    print(f"### Skipping rectangle: {rect}")
                    continue # Likely a rectangle enclosing the entire image.
                elif area_coverage_ratio < min_cover_ratio: # If the area has less than 98% white pixels, the rectangle probably encloses multiple keys.
                    # sub_rectangles = fit_rectangles(image, rect, min_cover_ratio, 10)
                    sub_rectangles = fit_rectangles(image, rect, min_cover_ratio, 2)
                    sr = [tuple(int(a) for a in r) for r in sub_rectangles]
                    rectangles = rectangles + sr
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