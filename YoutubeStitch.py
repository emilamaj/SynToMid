import cv2
import numpy as np
from pytube import YouTube
import argparse
import os
import time

# Download if not already downloaded. (in 480p at most)
def download_video(url):
    video_id = url.split("watch?v=")[-1]
    video_filename = f"video_{video_id}.mp4"
    video_folder = "videos"

    if not os.path.exists(video_folder):
        print("Creating video folder...")
        os.makedirs(video_folder)

    if not os.path.exists(os.path.join(video_folder, video_filename)):
        print("Downloading video...")
        yt = YouTube(url)
        yt.streams.filter(file_extension="mp4", resolution="480p").first().download(output_path=video_folder, filename=video_filename)
    else:
        print("Video already downloaded")

    return os.path.join(video_folder, video_filename)

# Find the amount in pixels that the frame needs to be shifted to align with the stitched image
def find_shift(frame, stitched_image):
    height = frame.shape[0]
    max_shift = height // 2
    best_shift = 0
    best_score = -np.inf

    method = cv2.TM_CCOEFF_NORMED
    for shift in range(max_shift):
        common_height = height - shift
        result = cv2.matchTemplate(frame[-common_height:], stitched_image[:common_height], method)
        _, score, _, _ = cv2.minMaxLoc(result)
        if score > best_score:
            best_score = score
            best_shift = shift

    # print(f"Best shift: {best_shift}, best score: {best_score}, max shift: {max_shift}")
    return best_shift

# Blends the part of the frame the is still in common with the stitched image (removes artefacts)
def blend_images(frame, stitched_image, shift):
    # An image is represented as a 3D array of shape (height, width, channels), the axis starting from the top left corner.
    alpha = 0.5
    common_height = frame.shape[0] - shift
    # print(f"common_height: {common_height}")
    # print(f"shift: {shift}")
    # print(f"frame.shape: {frame.shape}")
    # print(f"stitched_image.shape: {stitched_image.shape}")
    # print(f"Dimensions of operands: {stitched_image[:common_height].shape}, {frame[shift:common_height].shape}")
    stitched_image[:common_height] = cv2.addWeighted(stitched_image[:common_height], alpha, frame[shift:], 1 - alpha, 0)

# Pile the new part of the frame on top of the stitched image
def concatenate(frame, stitched_image, shift):
    new_part = frame[:shift]
    stitched_image = np.concatenate((new_part, stitched_image), axis=0)
    return stitched_image

def save_image(stitched_image, output_file):
    cv2.imwrite(output_file, stitched_image)

def stitch_frames(video_path, height, interval, start_time, end_time):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps)) # Seek the start time
    ret, stitched_image = cap.read()
    stitched_image = stitched_image[:int(stitched_image.shape[0] * height), :]
    current_time = start_time + interval
    nextPct = 5 # Print progress every 5%

    while current_time <= end_time:
# NOTE: Using cap.set might not be the fastest way (maybe read every frame but process only those of interest)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(current_time * fps))
        ret, frame = cap.read()

        frame = frame[:int(frame.shape[0] * height), :] # Clip the frame to select only the top part
        shift = find_shift(frame, stitched_image)
        blend_images(frame, stitched_image, shift)
        stitched_image = concatenate(frame, stitched_image, shift)
        
        # Print every time progress of 5% is made
        currentPct = int(100*(current_time-start_time) / (end_time-start_time))
        if currentPct >= nextPct:
            nextPct += 5
            print(f"Processed frame at {current_time}/{end_time} seconds ({currentPct}%) shift: {shift}")
        current_time += interval

    return stitched_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="The URL of the YouTube video")
    parser.add_argument("height", type=str, help="The height percentage of the frame to process (e.g., '25%')")
    parser.add_argument("interval", type=float, help="The time interval between frames to process, in seconds")
    parser.add_argument("start_time", type=float, help="The start time of the video, in seconds")
    parser.add_argument("end_time", type=float, help="The end time of the video, in seconds")
    args = parser.parse_args()

    # Time the download
    t0 = time.time()
    video_path = download_video(args.url)
    t1 = time.time()
    print(f"Downloaded video in {t1 - t0:.2f} seconds")
    height = float(args.height.strip("%")) / 100
    stitched_image = stitch_frames(video_path, height, args.interval, args.start_time, args.end_time)
    t2 = time.time()
    print(f"Stitched frames in {t2 - t1:.2f} seconds")
    output_file = "output.png"
    save_image(stitched_image, output_file)
    t3 = time.time()
    print(f"Stitched image saved to {output_file} in {t3 - t2:.2f} seconds")

if __name__ == "__main__":
    main()

# To run the script, use the following command:
# python YoutubeStitch.py https://www.youtube.com/watch?v=Zj_psrTUW_w 25% 0.5 5 15
