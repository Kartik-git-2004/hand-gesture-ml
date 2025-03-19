import cv2
from cvzone.HandTrackingModule import HandDetector

# Constants
WEBCAM_WIDTH = 1280
WEBCAM_HEIGHT = 720
IMAGE_PATH = 'img_1.png'

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Error: Could not open webcam.")

cap.set(3, WEBCAM_WIDTH)  # Set width
cap.set(4, WEBCAM_HEIGHT)  # Set height

# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Load the overlay image
img1 = cv2.imread(IMAGE_PATH)
if img1 is None:
    raise FileNotFoundError(f"Error: '{IMAGE_PATH}' not found or could not be loaded.")

# Variables for scaling and positioning
start_dist = None
scale = 0
cx, cy = WEBCAM_WIDTH // 2, WEBCAM_HEIGHT // 2  # Center of the screen

while True:
    success, img = cap.read()
    if not success:
        print("Warning: Failed to read frame from webcam. Skipping frame.")
        continue

    # Detect hands
    hands, img = detector.findHands(img)

    # Check if two hands are detected
    if len(hands) == 2:
        fingers1 = detector.fingersUp(hands[0])
        fingers2 = detector.fingersUp(hands[1])

        # Check if both hands are in the "pinch" gesture (thumb and index finger up)
        if fingers1 == [1, 1, 0, 0, 0] and fingers2 == [1, 1, 0, 0, 0]:
            lmList1 = hands[0]['lmList']
            lmList2 = hands[1]['lmList']

            if len(lmList1) > 8 and len(lmList2) > 8:
                p1 = lmList1[8][:2]  # Index finger tip of first hand
                p2 = lmList2[8][:2]  # Index finger tip of second hand

                # Calculate initial distance between fingers
                if start_dist is None:
                    start_dist, _, _ = detector.findDistance(p1, p2, img)

                # Calculate current distance and scale
                current_dist, _, _ = detector.findDistance(p1, p2, img)
                scale = int((current_dist - start_dist) // 2)
                cx, cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2  # Center between fingers
    else:
        start_dist = None  # Reset scaling if less than two hands are detected

    # Resize the overlay image based on hand distance
    h1, w1, _ = img1.shape
    newH, newW = max(1, h1 + scale), max(1, w1 + scale)  # Ensure dimensions are at least 1
    img1_resized = cv2.resize(img1, (newW, newH))

    # Calculate position to overlay the image
    x1 = max(0, cx - newW // 2)
    y1 = max(0, cy - newH // 2)
    x2 = min(WEBCAM_WIDTH, cx + newW // 2)
    y2 = min(WEBCAM_HEIGHT, cy + newH // 2)

    # Overlay the resized image
    img[y1:y2, x1:x2] = img1_resized[:y2 - y1, :x2 - x1]

    # Display the result
    cv2.imshow("Hand Detection", img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

