import os
import cv2
import mediapipe as mp


# Function to process each frame and blur detected faces
def process_img(frame, face_detection):
    H, W, _ = frame.shape  # Get frame dimensions
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            # Convert bounding box coordinates from relative to absolute
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # Apply blur to the face region
            frame[y1:y1 + h, x1:x1 + w] = cv2.blur(frame[y1:y1 + h, x1:x1 + w], (50, 50))

    return frame


# Define output directory and ensure it exists
output_dir = "/Users/nirmalfernando/PycharmProjects/Face_detection-and-blurring/output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection

# Open the input video file
file_path = "/Users/nirmalfernando/PycharmProjects/Face_detection-and-blurring/Portrait_Video_Footage.mp4"
cap = cv2.VideoCapture(file_path)

if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # Codec for MP4 format

# Create VideoWriter for output video
output_path = os.path.join(output_dir, 'output.mp4')
output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process video frames
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame to blur faces
        frame = process_img(frame, face_detection)

        # Write the processed frame to the output video
        output_video.write(frame)

# Release resources
cap.release()
output_video.release()

print(f"Processed video saved at: {"/Users/nirmalfernando/PycharmProjects/Face_detection-and-blurring/output"}")
