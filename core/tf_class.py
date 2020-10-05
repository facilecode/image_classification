import numpy as np 
import tensorflow as tf 
from PIL import Image, ImageOps
from core import debugger 
import cv2

class Tester:

    def __init__(self, model_path):

        self.model = tf.lite.Interpreter(model_path=model_path)
        self.model.allocate_tensors()

        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()
    
    @debugger.timeit
    def infer(self, data):
        self.model.set_tensor(self.input_details[0]['index'], data)
        self.model.invoke()
        return self.model.get_tensor(self.output_details[0]['index'])
    
    def camera(self):

        cam = cv2.VideoCapture(0)
        data = np.ndarray(shape=(1,224,224,3), dtype=np.float32)

        while True:
            ok, im = cam.read()

            im = Image.fromarray(im)
            im = ImageOps.fit(im, (224,224))
            image_array = np.asarray(im)
            norm_im = image_array
            #norm_im = (image_array.astype(np.float32)/127.0) - 1
            data[0] = norm_im  
            
            output_data  = self.infer(data)
            #print(output_data)  

    def predict(self, images, q):

        data = np.ndarray(shape=(1,224,224,3), dtype=np.float32)

        for img_path in images:

            print("Predicting image -> ", img_path)

            im = Image.open(img_path)
            im = ImageOps.fit(im, (224,224))
            image_array = np.asarray(im)
            norm_im = image_array
            #norm_im = (image_array.astype(np.float32)/127.0) - 1
            data[0] = norm_im  

            """
            self.model.set_tensor(self.input_details[0]['index'], data)
            self.model.invoke()
            output_data = self.model.get_tensor(self.output_details[0]['index'])
            """
            output_data  = self.infer(data)
            print(output_data)

            """
            #input_shape = self.input_details[0]['shape']
            #input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
            #print(input_data.shape)
            self.model.set_tensor(self.input_details[0]['index'], input_data)
            self.model.invoke()

            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            """ 

        