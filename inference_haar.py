from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
#from keras.optimizers import Adam

import cv2, numpy as np
#import pickle
from keras import backend as K
K.set_image_dim_ordering('th')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def fer(weights_path=None):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape = (1, 48, 48)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides = (1,1)))
    
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides = (1,1)))
    
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides = (1,1)))

    model.add(Flatten())
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(7, activation = 'softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model




if __name__ == "__main__":
    
    my_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    model = fer('3_model.h5')
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    cap = cv2.VideoCapture(0)
        # Enforce size of frames
    cap.set(3, 320) 
    cap.set(4, 240)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.namedWindow('res', cv2.WINDOW_NORMAL)
    img_in = np.zeros((240,320,1), np.uint8)
    img_rs = np.zeros((48,48,1), np.uint8)
    shot_id = 0
     
        #### Start video stream and online prediction
    while (True):
         # Capture frame-by-frame
    
    #        start_time = time.clock()
        
        ret, img = cap.read()
        #img = cv2.imread('skr_ak.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x-int(h/4),y-int(h/2.5)),(x+w+int(h/4),y+h+int(h/5)),(255,0,0),2)
            roi_gray = gray[y-int(h/2.5):y+h+int(h/5), x-int(h/4):x+w+int(h/4)]
            roi_color = img[y-int(h/2.5):y+h+int(h/5), x-int(h/4):x+w+int(h/4)]
            img_in = roi_gray
#            print img_in.shape
            if (img_in.shape[0] > 0 and img_in.shape[1] > 0):
                img_rs = cv2.resize(img_in,(48,48))
            else:
                img_rs = np.zeros((48,48,1), np.uint8)
            #eyes = eye_cascade.detectMultiScale(roi_gray)
            #for (ex,ey,ew,eh) in eyes:
                #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#        print img_rs.shape
        img_rs = np.array(img_rs).reshape(1, 1, 48, 48)
        out = model.predict(img_rs)
        #print np.argmax(out)
        #print my_list[np.argmax(out)]
        
        # we need to keep in mind aspect ratio so the image does
        # not look skewed or distorted -- therefore, we calculate
        # the ratio of the new image to the old image
        #r = 100.0 / frame.shape[1]
        dim = (352, 240)
 
        # perform the actual resizing of the image and show it
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        
        # Write some Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(resized,my_list[np.argmax(out)],(20,200), font, 1, (255,255,255),1,1)
        # Display the resulting frame
        cv2.imshow('Facial Expression Recognition',resized)        
#        cv2.imshow('img',img)
#        cv2.imshow('res',img_rs)
#        cv2.imwrite('fd_result.jpg',img)
#        cv2.imwrite('fd_result2.jpg', img_rs)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cv2.imwrite('fd_result2.jpg', resized)
    cap.release()
    cv2.destroyAllWindows()