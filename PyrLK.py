import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import warnings
import random
warnings.filterwarnings('ignore')
# Load the video file
cap = cv2.VideoCapture('Drone_Simulation2.mp4')
# Get the width and height of the video frames
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter object to write the output video

# Initialize variables for tracking the position of the drone
x = 0
y = 0
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize the previous points for optical flow calculation
prev_pts = None
prev_gray = None


# Initialize an empty list to store the position data
position_data = []

# start_time = 20
# end_time = 40

# Calculate the starting and ending frames based on the video's framerate
# fps = cap.get(cv2.CAP_PROP_FPS)
# start_frame = int(start_time * fps)
# end_frame = int(end_time * fps)

# Set the starting frame for the video capture
# cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Create a VideoWriter object to write the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (width, height))
# out = cv2.VideoWriter('output.mp4', fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))



# Process the video frame by frame
while True:
    # Read the current frame
    ret, frame = cap.read()

    # Check if there are no more frames
    if not ret:
        break
        # Check if there are no more frames or if we've reached the end frame
    # if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) >= end_frame:
    #     break
    # Resize the frame to 50% of its original resolution
    # frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Get the current time and calculate the elapsed time since the start
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    elapsed_time = current_time if len(position_data) == 0 else current_time - position_data[0][2]

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_pts is not None:
        # Get the optical flow for each point in the previous frame
        pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
        

               
        
        # Calculate the mean optical flow vector
        mean_flow = np.mean(pts[status == 1] - prev_pts[status == 1], axis=0)
        
        # Update the drone position based on the mean flow vector
        x += -mean_flow[0]
        y += mean_flow[1]

    # Detect good features to track in the current frame
    pts = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)

    # Update the previous frame and points
    prev_gray = gray.copy()
    prev_pts = pts



        # Add text to display the estimated position of the drone and elapsed time
    cv2.putText(frame, f'X: {x:.2f}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Y: {y:.2f}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Time: {elapsed_time:.2f}s', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Append the position data to the list
    position_data.append([x, y, current_time])

    # Convert the position data to a pandas DataFrame
    df = pd.DataFrame(position_data, columns=['x', 'y', 'time'])

            # Plot the x and y position data
    fig, ax = plt.subplots()
    # Get the start and end positions of the arrow
    arrow_start = (df['x'].iloc[0], df['y'].iloc[0])
    arrow_end = (df['x'].iloc[-1], df['y'].iloc[-1])
    

    ax.plot(df['x'], df['y'], linestyle='--',color='black')
    # Create a new figure to display the plot
    plt.figure()
    # Set the plot title and labels
    ax.set_title('Drone Position', fontsize=14)
    ax.set_xlabel('X position (pixels)', fontsize=12)
    ax.set_ylabel('Y position (pixels)', fontsize=12)
    ax.title.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    # Set the plot limits
    ax.set_xlim([df['x'].min() - 2000, df['x'].max() + 2000])
    ax.set_ylim([df['y'].min() - 2000, df['y'].max() + 2000])
    ax.set_position([0.75, 0.7, 0.2, 0.2])

    # Make the plot background transparent
    ax.patch.set_alpha(0.5)
    ax.patch.set_facecolor('none')

    # Create a new figure to display the plot
    plt.figure()

    # Convert the plot to an image
    fig.canvas.draw()
    plot_img = np.array(fig.canvas.renderer.buffer_rgba())

    # Close the plot
    plt.close()

    # Convert the image to BGR format for use in OpenCV
    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)

    # Resize the plot image to match the size of the video frame
    plot_img = cv2.resize(plot_img, (frame.shape[1], frame.shape[0]))

    # Overlay the plot image on the video frame
    cv2.addWeighted(plot_img, 0.4, frame, 1, 0, frame)
    if pts is not None:
        random_indices = random.sample(range(len(pts)), min(len(pts), 10))
        for i in random_indices:
            pt = pts[i]
            a, b = pt.ravel()
            cv2.circle(frame, (int(a), int(b)), 10, (0, 255, 0), -1)
    # Display the final frame
    
    # Write the frame to the output video
    out.write(frame)
    cv2.imshow('frame', frame)


    # Wait for the user to press a key
    if cv2.waitKey(1) == ord('q'):
        break
# Release the video file and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()