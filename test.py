import argparse
#import cv2
import torch
import torch.nn as nn
from glob import glob

parser = argparse.ArgumentParser()

parser.add_argument("-webcam", action="store_true")

parser.add_argument("-pb", action="store_true") # for keras
parser.add_argument("-full", action="store_true") # for torch
parser.add_argument("-framework", type=str, required=True)
parser.add_argument("-dataset", type=str, required=False)
parser.add_argument("-model", type=str, required=True)
#parser.add_argument("-path", action="store_true")

args = parser.parse_args()

def main():
    images = [
        "datasets/env/original/build/148.jpg",
        "datasets/env/original/car/ban3.jpg",
        "datasets/env/original/ped/image0.jpg",
        "datasets/env/original/tree/1.jpg",
        "datasets/env/original/trot/5282.jpg"
    ]

    if args.framework == "torch":
        from core import pytorch_class
        m = pytorch_class.Tester(model_path=args.model, full=args.full)

        m.predict_path(images)
        #m.train()
    if args.framework == "keras":
        from core import keras_class
        m = keras_class.Tester(model_path=args.model, pb=args.pb)

        m.predict_path(images)
        m.camera()
    
    if args.framework == "tflite":
        from core import tf_class
        m = tf_class.Tester(model_path=args.model)

        m.predict(images, False)
        m.camera()

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()

"""
python test.py -framework torch -model models/torch/env.json 
python test.py -framework keras -model models/keras/env.json
python test.py -framework tflite -model models/tflite/env.tflite
"""