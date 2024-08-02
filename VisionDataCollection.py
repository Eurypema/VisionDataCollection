# VisionDataCollection.py

import cv2 # Import the OpenCV library for computer vision tasks
import mediapipe.python.solutions.hands as mp_hands # Import MediaPipe Hands solution for hand tracking
import mediapipe.python.solutions.drawing_utils as drawing # Import drawing utilities from MediaPipe
import mediapipe.python.solutions.drawing_styles as drawing_styles # Import drawing styles from MediaPipe
import torchvision.transforms as transforms # Provides image transformations like resizing, normalization, and converting images to tensors
import os
from PIL import Image # Import the Python Imaging Library for image handling
import numpy as np
import threading
import time

transform = transforms.Compose([ # Composition of transformations
    transforms.Resize((224, 224)), # 224x224 is standard input size for pre-trained neural networks
    transforms.ToTensor(), # Converts to tensor (multidimensional array)

    # Normalizes based on mean and standard deviation of RGB channels
    transforms.Normalize(mean=[141.471, 128.245, 125.105], std=[65.856, 53.703, 51.865]) # Values based on what was calculated in RGBCalibrator 
])

# Flag to check if saving is in progress
saving_in_progress = threading.Event()

# Lock to prevent thread modification while being copied for saving
buffer_lock = threading.Lock()
buffer = []

def save_buffer_to_file(buffer, output_file_path):
    # Main function to save the buffer to the specified file path.

    def save_task(buffer_copy, output_file_path):
        # Nested function to handle the save operation in a separate thread.

        if saving_in_progress.is_set():
            print("A save operation is already in progress. Skipping this save.")
            return
        # Check if another save operation is in progress. If yes, skip this save.

        saving_in_progress.set()
        # Indicate that a save operation is now in progress.

        try:
            temp_file_path = output_file_path + ".tmp"
            combined_data = np.array(buffer_copy)
            # Create a temporary file path and convert the buffer into a NumPy array.

            if os.path.exists(output_file_path):
                try:
                    print("Preexisting file found, loading...")
                    existing_data = np.load(output_file_path, allow_pickle=True)
                    print("Existing data shape:", existing_data.shape)
                    print("Buffer data shape:", combined_data.shape)
                    # If the file exists, load the existing data.

                    if len(existing_data.shape) == len(combined_data.shape):
                        combined_data = np.concatenate((existing_data, combined_data))
                    else:
                        if len(existing_data.shape) > len(combined_data.shape):
                            combined_data = np.expand_dims(combined_data, axis=0)
                        combined_data = np.concatenate((existing_data, combined_data))
                    # Concatenate the existing data with the new data from the buffer.
                except Exception as e:
                    print(f"Error loading existing data: {e}")
                    combined_data = np.array(buffer_copy)
                    # Handle errors while loading existing data.

            try:
                with open(temp_file_path, 'wb') as f:
                    np.save(f, combined_data)
                    f.flush()
                    os.fsync(f.fileno())
                print(f"Temporary file saved at: {temp_file_path}")
                # Save the data to a temporary file first.
                # Opens a file at the path `temp_file_path` in binary write mode ('wb').
                # The 'with' statement ensures that the file is properly closed after writing, even if an error occurs.

                # Uses `np.save` to save the NumPy array `combined_data` to the file. This function serializes the array
                # and writes it in a binary format that is specific to NumPy, which includes metadata about the array's shape and data type.

                # The `f.flush()` call forces Python to flush the file's internal buffer, writing all data to the operating system's buffer.
                # This step helps ensure that all data intended to be written is actually sent to the OS.

                # The `os.fsync(f.fileno())` call forces the operating system to flush its buffers, writing the data to the disk.
                # This is a critical step to ensure data integrity, as it makes sure that the data is actually saved on the disk and
                # not just held in memory, preventing data loss in case of a sudden power failure or crash.

            except Exception as e:
                print(f"Error saving temporary file: {e}")
                return
                # Handle errors while saving the temporary file.

            if os.path.exists(temp_file_path):
                try:
                    os.replace(temp_file_path, output_file_path)
                    print(f"File renamed from {temp_file_path} to {output_file_path}")
                    # Atomically replace the old file with the new one.
                except Exception as e:
                    print(f"Error renaming temporary file: {e}")
                    # Handle errors during the renaming process.
            else:
                print(f"Temporary file not found after delay: {temp_file_path}")
                # If the temporary file is not found, log an error message.

        except Exception as e:
            print(f"Error during save: {e}")
            # Handle any other errors during the save process.
        finally:
            buffer_copy.clear()
            print("Buffer cleared and data saved.")
            saving_in_progress.clear()
            # Clear the buffer and release the save in progress flag.

    if not saving_in_progress.is_set():
        with buffer_lock:
            buffer_copy = buffer.copy()
            buffer.clear()
        saving_thread = threading.Thread(target=save_task, args=(buffer_copy, output_file_path))
        saving_thread.start()
        # time.sleep(0.3)
        saving_thread.join()
        time.sleep(1)
        # If no save is in progress, create a copy of the buffer and start a new thread for the save task.

    
def collect_data():

    # Initialize MediaPipe Hands
    hands = mp_hands.Hands(
        static_image_mode=False, # Process video stream
        max_num_hands=2, # Max number of hands to detect
        min_detection_confidence=0.5 # Minimum confidence for hand detection
    )
    
    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:

        # Ask the user which gesture to record
        gesture = input("Enter the gesture name to record (or 'exit' to quit): ").strip().lower()
        
        if '&' in gesture or 'c:/' in gesture or 'python.exe' in gesture:
            print("Invalid input detected. Please enter a valid gesture name.")
            continue
        
        elif gesture == '':
            print("Gesture name cannot be empty. Please enter a valid name.")
            continue

        elif gesture == 'exit':
            break

        print(f"Prepare to perform the gesture: {gesture}. Press 's' to start recording.")

        # Wait for the 's' key to start recording
        while True:
            _, frame = cap.read() # Read webcam frame
            cv2.imshow('frame', frame) # Display frame
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'): # Check if 's' key is pressed
                break

        buffer = [] # Buffer for batch writing
        buffer_size = 5000 # Define how often to write data to the file
        # frame_skip = 5  # Number of frames to skip
        # frame_count = 0

        print(f"Recording {gesture}. Press 'e' to end recording.")

        notification=False
        
        while True:
            ret, frame = cap.read() # Read frames
            if not ret: # Continue if the frame is not read successfully
                continue
            
            # frame_count += 1
            # if frame_count % frame_skip != 0:
            #     continue

            # Convert the frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame for hand detection
            results = hands.process(frame_rgb)

            # Draw hand annotations if hand landmarks are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    drawing.draw_landmarks(
                        frame, # Frame to draw on
                        hand_landmarks, # Detected hand landmarks
                        mp_hands.HAND_CONNECTIONS, # Connections between landmarks
                        drawing_styles.get_default_hand_landmarks_style(), # Default style for landmarks
                        drawing_styles.get_default_hand_connections_style() # Default style for connections
                    )
            
            # Display the annotated frame
            cv2.imshow('frame', frame)

            # Convert the frame from BGR (OpenCV format) to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert the RGB frame to PIL image
            pil_image = Image.fromarray(rgb_frame) 

            # Apply the transformations
            transformed_image = transform(pil_image)

            # Convert the transformed image tensor to a numpy array for saving
            data_array = np.array(transformed_image) # Converting to np.array because saving it as a List is extremely resource-intensive
            buffer.append(data_array)

            # Write buffer to file if it reaches the buffer size
            if len(buffer) >= buffer_size and notification is False:
                print("Buffer size has exceeded 5000. Press 'a' to save.")
                notification = True
                # print("Reclearing...")
                # buffer.clear()

            if cv2.waitKey(1) & 0xFF == ord('e'): # Check if 'e' is pressed to end recording
                print("Saving and exiting...")
                output_file_path = os.path.join('H:\\Training Data', f"{gesture}.npy")
                save_buffer_to_file(buffer, output_file_path)
                print("Successfully saved and exited.")
                break

            if cv2.waitKey(1) & 0xFF == ord('a'):
                print("Saving...")
                output_file_path = os.path.join('H:\\Training Data', f"{gesture}.npy")
                save_buffer_to_file(buffer, output_file_path)
                notification= False

        print(f"Data collected for gesture: {gesture}")
    
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function to collect data
collect_data()
