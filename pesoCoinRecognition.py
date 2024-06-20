import cv2
import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model('models/coinsmodel.h5')


cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


img_height, img_width = 256, 256


class_labels = ['1 peso', '10', '5']


predict_function = model.make_predict_function()

def preprocess_frame(frame):
    """Preprocess the frame for model prediction."""
    img = cv2.resize(frame, (img_height, img_width))
    img = img / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0) 
    return img

try:
 
    square_x, square_y, square_size = 0, 0, 0
    
    while True:
      
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to capture image")
            break
        
        preprocessed_img = preprocess_frame(frame)
        
        prediction = model.predict(preprocessed_img)
        
     
        coin_type = 'No Coin Detected'
        predicted_class_index = np.argmax(prediction)
        if prediction[0][predicted_class_index] > 0.8:
            coin_type = class_labels[predicted_class_index]
            
            
            coin_x, coin_y, coin_w, coin_h = 100, 100, 50, 50
            
           
            square_x = coin_x - 10  
            square_y = coin_y - 10
            square_size = max(coin_w, coin_h) + 20 
        
        cv2.rectangle(frame, (square_x, square_y), (square_x + square_size, square_y + square_size), (0, 255, 0), 2)
        
        
        cv2.putText(frame, coin_type, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
   
        cv2.imshow('Coin Detector', frame)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    
    cap.release()
    cv2.destroyAllWindows()