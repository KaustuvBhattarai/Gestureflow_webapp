from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

app = Flask(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image to detect hands
            results = hands.process(rgb_frame)

            # If hands are detected, draw landmarks, connections, and coordinates
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Iterate through each landmark to get its coordinates
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        # Get the coordinates in pixels
                        h, w, _ = frame.shape
                        cx, cy = int(landmark.x * w), int(landmark.y * h)

                        # Display the coordinates on the frame
                        cv2.putText(frame, f'{idx}: ({cx}, {cy})', 
                                    (cx, cy), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.4, (0, 255, 0), 1, cv2.LINE_AA)

            # Encode the frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame to display it in the live feed
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
