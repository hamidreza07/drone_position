import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import warnings
from math import radians, tan, log, cos
from scipy.interpolate import interp1d
import time
warnings.filterwarnings('ignore')
import cv2
from cv2 import SIFT_create,ORB_create,BRISK_create,AKAZE_create



def haversine(lat1, lon1, lat2, lon2):
    """
    Calculates the distance between two points on the Earth's surface using the Haversine formula.

    Args:
        lat1 (float): Latitude of the first point in degrees.
        lon1 (float): Longitude of the first point in degrees.
        lat2 (float): Latitude of the second point in degrees.
        lon2 (float): Longitude of the second point in degrees.

    Returns:
        float: The distance between the two points in kilometers.

    Example:
        >>> distance = haversine(40.7128, -74.0060, 37.7749, -122.4194)
        >>> print(distance)
        4124.84

    Notes:
        The Haversine formula calculates the distance between two points on the Earth's surface
        based on their latitude and longitude coordinates. It assumes a spherical Earth with a
        radius of 6371 kilometers. The result is an approximation and may have a slight error.

    """
    r = 6371  # radius of the Earth in kilometers
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = r * c
    return d


def convert_data(path):
    """
    Converts geographic data in an Excel file to Cartesian coordinates.

    Args:
        path (str): The path to the Excel file containing the geographic data.

    Returns:
        pandas.DataFrame: A DataFrame with Cartesian coordinates calculated from the geographic data.
                          The DataFrame includes additional columns 'x_distance' and 'y_distance'
                          representing the distances (in meters) from the first point to each point
                          in the DataFrame.

    Example:
        >>> data = convert_data('path/to/data.xlsx')
        >>> print(data.head())
              Latitude  Longitude  x_distance  y_distance
        0    40.7128   -74.0060    0.000000    0.000000
        1    37.7749   -122.4194  412118.972  526177.293
        2    34.0522   -118.2437  362346.800  437139.521
        ...

    Notes:
        This function internally uses the Haversine formula to calculate the distance between two
        pairs of latitude and longitude coordinates.

    """
    # Define the Haversine function



    # Load the data into a pandas DataFrame
    data = pd.read_excel(path)
    # Calculate the distance from the first point to each point in the DataFrame
    lat1 = data['Latitude'].iloc[0]
    lon1 = data['Longitude'].iloc[0]
    data['x_distance'] = data['Latitude'].apply(lambda lat2: haversine(lat1, lon1, lat2, lon1)*1000)
    data['y_distance'] = data['Longitude'].apply(lambda lon2: haversine(lat1, lon1, lat1, lon2)*1000)
    return data


def calculate_black_frame(frame, min_black_percentage):
    """
    Calculates the percentage of black pixels in a frame and checks if it exceeds the specified threshold.

    Args:
        frame (numpy.ndarray): The input frame for which black pixel percentage needs to be calculated.
        min_black_percentage (int): The minimum percentage of black pixels for the frame to be considered black.

    Returns:
        bool: True if the black percentage of the frame is equal to or greater than the specified threshold, False otherwise.
    """
    total_pixels = frame.size
    black_pixels = np.sum(frame == 0)
    black_percentage = (black_pixels / total_pixels) * 100
    return black_percentage >= min_black_percentage
    

def drone_local(video_path: str, method: str, scale: float, rotation_angle: int, params: dict, imu_path: str, min_black_percentage: int, interval: int,feature_method:str, feature_param: dict, visual_mode: bool = False):
    """
    Performs localization of a drone using optical flow and IMU data.

    Args:
        video_path (str): The path to the video file containing the drone footage.
        method (str): The method to use for optical flow calculation ('FlowPyrLK' or 'Farneback').
        scale (float): The scale factor to convert optical flow from pixel to meter.
        rotation_angle (int): The rotation angle of the drone's optical flow sensor in degrees.
        params (dict): Parameters for the optical flow algorithm.
        imu_path (str): The path to the IMU data file containing the drone's position information.
        min_black_percentage (int): The minimum percentage of black pixels in a frame for it to be considered valid.
        interval (int): The frame interval for optical flow calculation.
        feature_detection_param (dict): Parameters for the feature detection algorithm.
        visual_mode (bool, optional): Whether to enable visual mode to display the drone's position. Defaults to False.

    Returns:
        None

    Example:
        drone_local('path/to/video.mp4', 'FlowPyrLK', 0.1, 45, {'winSize': (15, 15), 'maxLevel': 2, 'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}, 'path/to/imu_data.csv', 20, 10, {'maxCorners': 100, 'qualityLevel': 0.3, 'minDistance': 7, 'blockSize': 7}, visual_mode=True)
    """
    t1 = time.process_time()
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    prev_pts = None
    x = x_cal = x_new = 0
    y = y_cal = y_new = 0
    real_position = []
    optical_position = []
    distance_diff_values = []
    data = convert_data(imu_path)


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if visual_mode:
        out = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 4 / 3), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    else:
        out = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / interval)
    while True:
        for _ in range(frame_interval):
            ret, frame = cap.read()
            if not ret:
                break
        if not ret:
            break



        if calculate_black_frame(frame,min_black_percentage):
            # Skip the corrupted frame
            continue

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_pts is not None:
            if method == 'FlowPyrLK':                   
                    # Perform optical flow calculation using the keypoints
                pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **params)

                # Calculate the mean optical flow vector
                mean_flow = np.mean(pts[status == 1] - prev_pts[status == 1], axis=0)
                # Convert the optical flow values from pixel to meter
   
            elif method == "Farneback":
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **params)
                mean_flow = np.mean(flow, axis=(0,1))
                
                
            x -= mean_flow[0]*scale
            y += mean_flow[1]*scale
                        

            
            
        optical_position.append([x, y, current_time])
        # Convert the position data to a pandas DataFrame
        df_optical = pd.DataFrame(optical_position, columns=['x', 'y', 'time'])
        
        if feature_method == 'SIFT':
            # Initialize the SIFT detector
            sift = SIFT_create(**feature_param)
            keypoints, descriptors = sift.detectAndCompute(gray, None)

            # Convert keypoints to numpy array format
            pts = np.array([keypoint.pt for keypoint in keypoints], dtype=np.float32).reshape(-1, 1, 2)
        elif feature_method == 'ORB':
            # Initialize the ORB detector
            orb = ORB_create(**feature_param)
            keypoints, descriptors = orb.detectAndCompute(gray, None)

            # Convert keypoints to numpy array format
            pts = np.array([keypoint.pt for keypoint in keypoints], dtype=np.float32).reshape(-1, 1, 2)
            
            
            
        elif feature_method == 'BRISK':
            # Initialize the ORB detector
            BRISK = BRISK_create(**feature_param)
            keypoints, descriptors = BRISK.detectAndCompute(gray, None)

            # Convert keypoints to numpy array format
            pts = np.array([keypoint.pt for keypoint in keypoints], dtype=np.float32).reshape(-1, 1, 2)   
            
               
        elif feature_method == 'AKAZE':
            # Initialize the ORB detector
            AKAZE = AKAZE_create(**feature_param)
            keypoints, descriptors = AKAZE.detectAndCompute(gray, None)

            # Convert keypoints to numpy array format
            pts = np.array([keypoint.pt for keypoint in keypoints], dtype=np.float32).reshape(-1, 1, 2)      
            
            
                  
        elif feature_method == 'goodFeaturesToTrack':
            
            pts = cv2.goodFeaturesToTrack(gray, **feature_param)
            

            

        elif feature_method == 'FastFeatureDetector':
                fast = cv2.FastFeatureDetector_create(**feature_param)
                kp = fast.detect(gray, None)
                pts = cv2.KeyPoint_convert(kp) 
        else:
                break
        # Update the previous frame and points
        prev_gray = gray.copy()
        prev_pts = pts
        distance_opt = np.sqrt(int(x) ** 2 + int(y) ** 2)
        distance = np.sqrt(int(x_new) ** 2 + int(y_new) ** 2)

        if distance == 0:
            distance_difference = 0
        else:
            distance_difference = abs((distance - distance_opt) / distance) * 100

        distance_diff_values.append([distance_difference, current_time])

        

        for index, row in data.iterrows():
            if index + 1 >= len(data):
                break
            elif current_time == row['second']:
                theta = math.radians(rotation_angle)

                # Convert the optical flow values from pixel to meter
                x_new = row['x_distance'] * math.cos(theta) - row['y_distance'] * math.sin(theta)
                y_new = row['x_distance'] * math.sin(theta) + row['y_distance'] * math.cos(theta)

            elif row['second'] < current_time < data.iloc[index + 1]['second']:
                x_cal = np.interp(current_time, [row['second'], data.iloc[index + 1]['second']], [row['x_distance'], data.iloc[index + 1]['x_distance']])
                y_cal = np.interp(current_time, [row['second'], data.iloc[index + 1]['second']], [row['y_distance'], data.iloc[index + 1]['y_distance']])
                theta = math.radians(rotation_angle)

                # Convert the optical flow values from pixel to meter
                x_new = x_cal * math.cos(theta) - y_cal * math.sin(theta)
                y_new = x_cal * math.sin(theta) + y_cal * math.cos(theta)

            real_position.append([x_new, y_new, row['second']])
            df_real = pd.DataFrame(real_position, columns=['x_cal', 'y_cal', 'time_cal'])

        if visual_mode:
            cv2.putText(frame, f'X: {x:.2f} m       Y: {y:.2f} m', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f'X_act: {x_new:.2f} m  Y_act: {y_new:.2f} m', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f'distance optical: {distance_opt:.2f} m', (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'distance actual: {distance:.2f} m', (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'distance difference: {distance_difference:.2f}', (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)


            # Plot the x and y position data
            fig, ax = plt.subplots()
            ax.plot(df_optical['x'], df_optical['y'], color='red', linewidth=3)
            ax.plot(df_real['x_cal'], df_real['y_cal'], color='blue', linewidth=2)

            # Set the plot title and labels
            ax.set_title('Drone Position', fontsize=14, loc='center')
            ax.set_xlabel('X position (m)', fontsize=12, loc='center')
            ax.set_ylabel('Y position (m)', fontsize=12, loc='center')
            ax.title.set_color('black')
            ax.xaxis.label.set_color('black')
            ax.yaxis.label.set_color('black')

            # Set the plot limits
            ax.set_xlim([df_optical['x'].min() - 250, df_optical['x'].max() + 250])
            ax.set_ylim([df_optical['y'].min() - 250, df_optical['y'].max() + 250])

            # Make the plot background transparent
            ax.patch.set_alpha(0.5)
            ax.patch.set_facecolor('none')

            # Convert the plot to an image
            fig.canvas.draw()
            plot_img = np.array(fig.canvas.renderer.buffer_rgba())

            # Close the plot
            plt.close()

            # Convert the image to BGR format for use in OpenCV
            plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)

            # Resize the plot image to match the size of the video frame
            plot_img = cv2.resize(plot_img, (int(frame.shape[1] / 3), frame.shape[0]))

            # Create a new image that combines both the video frame and the map
            combined_img = np.concatenate((frame, plot_img), axis=1)
            # Wait for the user to press a key
            out.write(combined_img)

            if cv2.waitKey(1) == ord('q'):
                break
            cv2.imshow('frame', combined_img)
        if not visual_mode:
            out.write(frame)

    # Release the video capture and output video writer
    cap.release()
    out.release()
    t2 = time.process_time()
    print(f'total process time: {t2-t1}')
    print(f'loss: {distance_diff_values[-1][0]:.2f} ')
    print(f'final result :')
    print(df_optical.tail(10))
    # Close all OpenCV windows
    cv2.destroyAllWindows()
#   FlowPyrLK parameters
lk_params = dict(winSize=(30, 30), maxLevel=10, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.03))
#   goodFeaturesToTrack parameter
gft_param =dict(maxCorners = 100, qualityLevel =0.01, minDistance = 50)
#   Farneback parameters
params = dict(pyr_scale=0.5, levels=3, winsize=45, iterations=1, poly_n=3, poly_sigma=1.2, flags=0)
# FastFeatureDetector parameter
ft_param = dict(threshold=10, nonmaxSuppression=True)
# ORB paramameter
orb_params = dict(nfeatures=1000, scaleFactor=1.3, nlevels=5, edgeThreshold=10)
# sift parameter
sift_params = dict(nfeatures=1000, contrastThreshold=0.001, edgeThreshold=1000, sigma=2.6,nOctaveLayers=5)
# azake parameter
azake_params = dict(threshold=0.001, nOctaves=3,  nOctaveLayers=3,  descriptor_channels=3,  descriptor_size=0)
# BRISK paramameter
BRISK_params = dict(thresh=30, octaves=3, patternScale=1.0)




drone_local('IMG_6655.MP4',method='FlowPyrLK',scale=0.100916667 ,rotation_angle=25,params=lk_params,
            imu_path='video.xlsx',min_black_percentage=5,interval=3,feature_method='AKAZE'
            ,feature_param=azake_params,visual_mode=True)