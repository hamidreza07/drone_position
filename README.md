# Drone Position Estimation

This project is an implementation of a Python script that uses optical flow and feature detection techniques to estimate the position of a drone in a video. The script reads a video file, calculates the optical flow for each frame, and estimates the position of the drone based on the mean optical flow vector.

The script also creates a plot of the drone's position over time and saves the output video with the estimated position information displayed on each frame.


## Dependencies
- Python 3
- OpenCV
- NumPy
- Pandas
- Matplotlib

## Usage
1. Clone the repository or download the script.
2. Make sure the dependencies are installed.
3. Place the input video file in the same directory as the script.
4. Update the filename of the input video in the script (line 8).
5. Run the script.
The output video file will be saved as "output.mp4" in the same directory as the script. The plot of the drone's trajectory will be displayed on the screen.

### Notes
- The script uses the Lucas-Kanade optical flow method to track the drone's movement between frames.
- The position of the drone is estimated based on the mean optical flow vector of the tracked points.
- The plot of the drone's trajectory is generated using Matplotlib.
- The script can be modified to track other objects in a video by changing the tracking parameters and adjusting the plot accordingly.