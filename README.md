# SynToMid
This tool helps you convert synthesia-like piano videos of Youtube to a MIDI (.mid) file that can be read by most music tools. It makes extensive use of OpenCV to process the video frames, and to extract the notes played.

Currently implemented:
- YoutubeStitch.py: Stitch together the frames of the Youtube video into a tall png image of the keys pressed.
- WaterfallProcess.py: Convert the stitched image to a list of list of rectangles representing the keys pressed. **Very buggy, not yet usable for note extraction.**

Todo:
- [ ] Better processing of the notes, especially close ones. (Try again pseudo-gradient descent but with cost function in absolute pixels, like L=[10*black_pixels - 1*white_pixels], where the pixels proposed are the ones enclosed by a rectangle. Beware of the fact that close rectangles will mask each otehr, L also needs an overlap term.)
- [ ] Implement ReadNotes.py, which converts the processed rectangles to a list of notes.
- [ ] Fix YoutubeStitch.py so that the note timings are respected. (Find average scroll rate, and blindly stitch ? Scroll rate might not be constant...)

## YoutubeStitch.py
This is the tool that stitches together the frames of the Youtube video into a tall png image of the keys pressed.

![Example output of YoutubeStitch.py](/output_stitch.png)

This is the result of the script for the first 30 seconds of Chopin - Ballade No. 1 played by Rousseau (https://www.youtube.com/watch?v=Zj_psrTUW_w)
This project will consist in 3 parts:
- YoutubeStitch.py
- WaterfallProcess.py
- ReadNotes.py

# YoutubeStitch.py
Here is the usage for YoutubeStitch.py:
```bash
python YoutubeStitch.py <url> <height> <interval> <start> <stop>
```

- url (string): Url of the Youtube video
- height (string): Percent of the height of the video to process (starting from the top, to ignore the hands of the player)
- interval (float): Interval in seconds between the frames, to allow the script to run faster
- start (float): Start position of the video, in seconds (to ignore intros)
- stop (float): Stop position of the video, in seconds

The process can run for several minutes.
You can speedup the script by lowering the height processed. It also removes a lot of the visual artefacts that might remain at the end.
Reducing the height however completely messes up the duration of the long notes and the long silences.

It produces a file called `output.png`, which is the stitched image of the video.

# WaterfallProcess.py
This is the tool that converts the stitched image to a list of list of rectangles representing the keys pressed.
Usage:
```bash
python WaterfallProcess.py <input_png> <output_mid>
```
Here is the current state of the image processing done with OpenCV:

![Example output of WaterfallProcess.py](/output_process.png)