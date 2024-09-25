import cv2
import numpy as np

def process_frame(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and detail
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform edge detection using the Canny method
    edges = cv2.Canny(blur, 50, 150)
    
    # Define a region of interest (mask)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    
    # Focus on the lower half of the frame (where the lanes are)
    polygon = np.array([[
        (0, height),
        (width, height),
        (width //1.7, height//1.7 )
    ]], np.int32)
    
    cv2.fillPoly(mask, polygon, 255)
    
    # Mask the edges image to focus on the region of interest
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Use Hough Line Transform to detect lanes
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=40, maxLineGap=150)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    return frame

# Capture video from the camera (use Pi Camera or USB camera)
cap = cv2.VideoCapture("lane1.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    frame=cv2.resize(frame,(1100,700))
    if not ret:
        break
    
    # Process the frame for lane detection

    processed_frame = process_frame(frame)
    
    # Display the processed frame
    cv2.imshow("Lane Detection", processed_frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
