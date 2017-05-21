# This program takes in a video feed and predicts the facial expression using trained CNN model

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
#import time
#from keras.optimizers import Adam
import cv2, numpy as np
#import pickle
from keras import backend as K
K.set_image_dim_ordering('th')

# Load the OpenCV - Haar Classifier - xml file to localize faces in video feed
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load trained weights to the model
def fer(weights_path=None):
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

    return model




if __name__ == "__main__":

    # The list of facial expressions to be classified
    my_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # Call the function to load trained weights from .h5 file
    model = fer('3_model.h5')
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    # Capture the input video
    cap = cv2.VideoCapture(0)
        
    cap.set(3, 320) 
    cap.set(4, 240)
#    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
#    cv2.namedWindow('res', cv2.WINDOW_NORMAL)
    img_in = np.zeros((240,320,1), np.uint8)
    img_rs = np.zeros((48,48,1), np.uint8)
    shot_id = 0
     
        
    while (True):
         
        ret, img = cap.read()
        #img = cv2.imread('skr_ak.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Obtain the region in the video frame where the face was detected by Haar Classifier
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        
        X=0
        Y=0
        H=0
        W=0
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            img_in = roi_gray
         
            if (img_in.shape[0] > 0 and img_in.shape[1] > 0):
                img_rs = cv2.resize(img_in,(48,48))
            else:
                img_rs = np.zeros((48,48,1), np.uint8)
            
            X=x
            Y=y
            H=h
            W=w
            
            
#            time.sleep(0.1)
        img_rs = np.array(img_rs).reshape(1, 1, 48, 48)
        # Run the prediction on the face image
        out = model.predict(img_rs)
        dim = (352, 240)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)        
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(resized,my_list[np.argmax(out)],(X,Y+H+int(H/4)), font, 1, (255,255,255),1,1)       
        
        
        # Display output prediction
        cv2.imshow('Facial Expression Recognition',resized)        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
#    cv2.imwrite('fd_result2.jpg', resized)
    cap.release()
    cv2.destroyAllWindows()