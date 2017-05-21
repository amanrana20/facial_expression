# Imports
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization

import cv2, numpy as np

from keras import backend as K
K.set_image_dim_ordering('th')

class Facial_Expression_Recognition:
    
    def __init__(self):
        self.run()
    
    # Calling haar cascade classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


    # Defining the model
    def create_model(self, weights_path=None):
        
        model = Sequential()
        model.add(Convolution2D(64, 3, 3, input_shape = (1, 48, 48)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,2), strides = (2,2)))
        
        model.add(Convolution2D(128, 3, 3))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,2), strides = (2,2)))
        
        model.add(Convolution2D(256, 3, 3))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,2), strides = (2,2)))
        
        model.add(Convolution2D(512, 3, 3))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,2), strides = (2,2)))
        
        model.add(Flatten())
        model.add(Dense(1024, activation = 'relu'))
        model.add(Dense(7, activation = 'softmax'))

        if weights_path:
            model.load_weights(weights_path)
        
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        
        return model


    def predict_expression(self, model, image):
        return np.argmax(model.predict(image))

    def run(self):
        facial_expression_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Creating model and loading weights
        model = self.create_model('3_model.h5')
        
        # Starting video camera
        cap = cv2.VideoCapture(0)
        
        # Setting the videofeed size to 320, 240
        cap.set(3, 1000)
        cap.set(4, 600)
        
        
        while (True):
             
            ret, img = cap.read()
            
            # Converting the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Using Haar Cascade Classifier to detect multiple faces in each camera feed
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            dim = (1000, 600)
        
            # Stores the sub-frames containing the detected faces
            FACES_DETECTED = []
            
            # Looping around every face detected, extracting each face
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                img_in = roi_gray
             
                if (img_in.shape[0] > 0 and img_in.shape[1] > 0):
                    img_rs = cv2.resize(img_in,(48,48))
                else:
                    continue
                
                img_rs = np.array(img_rs).reshape(1, 1, 48, 48)
                FACES_DETECTED.append([img_rs, (h, w), (x, y)])

            face_predictions = []
            for face in FACES_DETECTED:
                face_predictions.append([self.predict_expression(model, face[0]), face[-2], face[-1]])
            
            font = cv2.FONT_HERSHEY_SIMPLEX

            for each_prediction in face_predictions:
                x, y = each_prediction[-1]
                h = each_prediction[-2][0]
                y += (5*h/4)
                cv2.putText(img,facial_expression_list[each_prediction[0]],(x,y), font, 1, (255, 255, 255), 1, 1)

            cv2.imshow('Facial Expression Recognition',img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    #    cv2.imwrite('fd_result2.jpg', resized)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    Facial_Expression_Recognition()
