import cv2 as cv
import numpy as np
from fer import FER
from PIL import Image
import os
import collections

class EmotionDetector:
    def __init__(self):
        self.cam = cv.VideoCapture(0)
        self.face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_detector = FER(mtcnn=True)
        self.dominant_emotion = None
        self.emoji_selector = EmotionEmoji()
        self.face_dimensions = collections.deque(maxlen=10)  # Store last 10 face dimensions
        self.smoothed_dimension = None

    def detect_emotion(self):
        ret, img = self.cam.read()
        if not ret:
            return None, None

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        emotions = self.emotion_detector.detect_emotions(img)
        
        emoji_img = img.copy()
        
        emoji = None
        self.dominant_emotion = None
        
        for (x, y, w, h) in faces:
            if emotions:
                emotion = emotions[0]['emotions']
                self.dominant_emotion = max(emotion, key=emotion.get)
                
                emoji = self.emoji_selector.get_emoji(self.dominant_emotion)
                emoji_img = self.overlay_emoji(emoji_img, emoji, x, y, w, h)

        if self.dominant_emotion:
            text = f"Detected: {self.dominant_emotion}"
            text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = emoji_img.shape[1] - text_size[0] - 10
            text_y = text_size[1] + 10
            cv.putText(emoji_img, text, (text_x, text_y), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv.imshow('Emoji Overlay', emoji_img)
        return self.dominant_emotion, emoji

    def overlay_emoji(self, img, emoji_path, x, y, w, h):
        emoji = Image.open(emoji_path)
        
        # Use smoothed dimension for emoji size
        emoji_size = self.smoothed_dimension if self.smoothed_dimension else min(w, h)
        
        emoji = emoji.resize((emoji_size, emoji_size), Image.LANCZOS)
        emoji = np.array(emoji)
        
        emoji_x = x + (w - emoji_size) // 2
        emoji_y = y + (h - emoji_size) // 2
        
        if emoji.shape[2] == 4:
            mask = emoji[:, :, 3] / 255.0
            emoji = emoji[:, :, :3]
        else:
            mask = np.ones((emoji_size, emoji_size))
        
        # Convert emoji from RGB to BGR
        emoji = cv.cvtColor(emoji, cv.COLOR_RGB2BGR)
        
        for c in range(3):
            img[emoji_y:emoji_y+emoji_size, emoji_x:emoji_x+emoji_size, c] = (
                emoji[:, :, c] * mask + 
                img[emoji_y:emoji_y+emoji_size, emoji_x:emoji_x+emoji_size, c] * (1 - mask)
            )
        
        return img

    def run(self):
        while True:
            self.detect_emotion()
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        self.cam.release()
        cv.destroyAllWindows()

    def get_dominant_emotion(self):
        return self.dominant_emotion


class EmotionEmoji:
    def __init__(self):
        emoji_dir = os.path.join('emoji_pngs')
        self.emotion_emoji_map = {
            'angry': os.path.join(emoji_dir, 'angry.png'),
            'disgust': os.path.join(emoji_dir, 'disgust.png'),
            'fear': os.path.join(emoji_dir, 'fear.png'),
            'happy': os.path.join(emoji_dir, 'happy.png'),
            'sad': os.path.join(emoji_dir, 'sad.png'),
            'surprise': os.path.join(emoji_dir, 'surprise.png'),
            'neutral': os.path.join(emoji_dir, 'neutral.png')
        }
        self.unknown_emoji = os.path.join(emoji_dir, 'unknown.png')

    def get_emoji(self, emotion):
        return self.emotion_emoji_map.get(emotion.lower(), self.unknown_emoji)

def main():
    print("Starting Emotion Detection...")
    detector = EmotionDetector()
    try:
        detector.run()
    except KeyboardInterrupt:
        print("\nEmotion Detection stopped by user.")
    finally:
        print("Cleaning up...")
        detector.cam.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()