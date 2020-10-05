import cv2
from sys import platform
from glob import glob
from PIL import Image, ImageOps
import numpy as np
import os
import shutil

class Creator:

    def __init__(self, save_path):
        
        self.platform = platform
        self.save_path = save_path
        self.devices = self.get_devices()
        self.main_device = None
        self.transforms = {
            "gray": None, 
            "resize": None, #(224,224)
            "crop": None, #[0,300, 0, 300],
            "rotate": None # by n*90 degrees
        }
        self.classes = ["air", "hand", "head", "couille"]
    
    def get_devices(self):
        # to-do
        devices = [] # according to os
        return devices
    
    def delete_classes(self, classes):

        for c in classes:
            shutil.rmtree(os.path.join(self.save_path, c))

    def add_classes(self, classes):
        """
        Creates folders for each class to store images
        """
        for c in classes:

            if os.path.exists(path := os.path.join(self.save_path, c)):
                print("Dir for ", c, " already exists")
            else:
                print("Creating for ", c)
                os.mkdir(path)

    def delete_image_from_class(self, c):

        path = os.path.join(self.save_path, c)
        images = glob(path)

        last = images[-1]
        os.remove(last)

    def delete_all_images(self, c):

        path = os.path.join(self.save_path, c)
        images = glob(path)

        for f in images:
            os.remove(f)

    def transform(self, im):
        """
        Server sends a dict with actual transformations
        {
            gray: True/False,
            resize: (x,y),
            crop: (y1,y2, x1,x2),
            rotate: a
        }
        """
        
        if self.transforms["gray"]:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
        
        if (size := self.transforms["resize"]):
            im = cv2.resize(im, size) 
        
        if (crop := self.transforms["crop"]):
            """
            cv2 crop routine via slicing

            im = im[y:y+h, x:x+w]
            """
            im = im[crop[0]:crop[1], crop[2]:crop[3]]

        
        if (rotate := self.transforms["rotate"]):
            """
            rotate n * 90 times
            """
            rotateCode = None

            if rotate > 0:
                rotateCode = cv2.ROTATE_90_CLOCKWISE
            else:
                rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE

            for _ in range(rotate):
                im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        
        return im
        
    def set_main_device(self, path):

        self.main_device = cv2.VideoCapture(path)
        ok, im = self.main_device.read()

        # check whether camera is returning frames
        if ok:
            print("Device set")
            return True
        else:
            print("Invalid device")
            return False

    def process_frame(self):

        ok, im = self.main_device.read()

        im = self.transform(im)

        cv2.imshow("image", im)
        cv2.imwrite(self.save_path+".jpg", im)
        cv2.waitKey(1)

    def __str__(self):
        return str(self.__dict__)
    
c = Creator("here")
c.set_main_device(0)

while True:
    c.process_frame()