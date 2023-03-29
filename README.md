## SynToMid
This tool helps you convert synthesia-like piano videos of Youtube to a MIDI (.mid) file that can be read by most music tools.
Currently, the only part that is implemented is the one that grabs the waterfall of keys and unrolls it into a long png image

# YoutubeStitch.py
This is the tool that stitches together the frames of the Youtube video into a tall png image of the keys pressed.

![Example output](/output.png)

This is the result of the script for the first 30 seconds of Chopin - Ballade No. 1 played by Rousseau (https://www.youtube.com/watch?v=Zj_psrTUW_w)

Here is the usage for YoutubeStitch.py:
python YoutubeStitch.py <url> <height> <interval> <start> <stop>
- url (string): Url of the Youtube video
- height (string): Percent of the height of the video to process (starting from the top, to ignore the hands of the player)
- interval (float): Interval in seconds between the frames, to allow the script to run faster
- start (float): Start position of the video, in seconds (to ignore intros)
- stop (float): Stop position of the video, in seconds

The process can run for several minutes.
You can speedup the script by lowering the height processed. It also removes a lot of the visual artefacts that might remain at the end.
Reducing the height however completely messes up the duration of the long notes and the long silences.