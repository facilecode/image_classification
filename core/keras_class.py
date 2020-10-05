import tensorflow as tf  
import tensorflow.keras as keras 
import tensorflow.keras.applications as models
import os
import glob
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps
import numpy as np
import json
import cv2
from core import debugger

class Trainer:

    models = {
        "resnet50": models.resnet50,
        "resnet50V2": models.ResNet50V2,
        "inceptionV3": models.inception_v3,
        "inceptionResNetV2": models.InceptionResNetV2,
        "mobilenetv2": models.MobileNetV2,
        "densenet121": models.DenseNet121,
        "resnet50": models.resnet50,
    }

    def __init__(self, dataset_path, epochs, batch, base_model, weights, model_name, gpu):

        self.model = None
        self.model_name = model_name
        self.classes = os.listdir(dataset_path)
        self.model_info = {
            "base_model": None,
            "h5": None,
            "pb": None,
            "classes": self.classes
        }
        self.transforms = {

        }
        #self.classes = [1,2,3,4,5]
        self.init_model(base_model="mobilenetv2", weights=weights, output_dim=len(self.classes))
        self.train(dataset_path=dataset_path, epochs=epochs, batch=batch, gpu=gpu)

    def init_model(self, base_model, weights, output_dim):

        if base_model == "mobilenetv2":
            base = models.MobileNetV2(
                alpha=1.0,
                include_top=False,
                input_shape=(224,224,3),
                weights=weights,
                #classes=output_dim,
                #classifier_activation="softmax"
            )

        if base_model == "resnet50":
            base = models.resnet50(
                include_top=False,
                weights=weights,
                input_shape=(224,224,3),
                #classes=output_dim,
                #classifier_activation="softmax"
            )

        inputs = keras.Input(shape=(224,224,3))
        x = base(inputs, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(output_dim, activation=keras.activations.softmax)(x)
        #outputs = keras.layers.Softmax()(x)
        self.model = keras.Model(inputs, outputs)

        #self.model.trainable = True
        
    def train(self, dataset_path, epochs, batch, gpu):
        
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            dataset_path,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(224,224),
            batch_size=4,
            shuffle=True
        )
        
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            dataset_path,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(224,224),
            batch_size=4,
            shuffle=True
        )

        """
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=(0.1,0.5),
            validation_split=0.2 # only one dir for both
        )

        train_generator = train_datagen.flow_from_directory(
            dataset_path,
            target_size=(224,224), # (299,299) for inception to-do
            batch_size=batch,
            subset="training",
            class_mode="sparse"
        )

        val_generator = train_datagen.flow_from_directory(
            dataset_path,
            target_size=(224,224),
            batch_size=batch,
            subset="validation",
            class_mode="sparse"
        )
        """

        #to-do add optimizers loss ...

        print(self.model.summary())

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.0001),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

        filepath = "models/keras/temp"
        best_save_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor="val_acc",
            mode="max",
            save_best_only=True,
            #save_weights_only=True
        )

        # without augm
        
        self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[best_save_callback]
        )
        
        """
        self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            #callbacks=[best_save_callback]
        )
        """
        loss, accuracy = self.model.evaluate(val_ds)
        print('Test accuracy :', accuracy)

        self.model_info["h5"] = "models/keras/env.h5"
        self.model_info["pb"] = "models/keras/env"

        with open("models/keras/" + self.model_name + ".json", "w") as f:
            json.dump(self.model_info, f)        

        #self.model.load_model(filepath)
        self.model.save("models/keras/env")
        self.model.save("models/keras/env.h5")

        # tflite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()

        with open("models/tflite/" + "env" + ".tflite", "wb") as f:
            f.write(tflite_model)

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model_q = converter.convert()

        with open("models/tflite/q_" + "env" + ".tflite", "wb") as f:
            f.write(tflite_model_q)
        
        converter.target_spec.supported_types = [tf.float16]
        tflite_model_q_16 = converter.convert()

        with open("models/tflite/q_16_" + "env" + ".tflite", "wb") as f:
            f.write(tflite_model_q)



class Tester:

    def __init__(self, model_path, pb):
        
        f = open(model_path)
        self.model_info = json.load(f)
        self.init_model(pb)
    
    def init_model(self, pb):

        if pb == True:
            self.model = keras.models.load_model(
                self.model_info["pb"]
            )
        
        if pb == False:
            self.model = keras.models.load_model(
                self.model_info["h5"]
            )            
    @debugger.timeit
    def infer(self, data):
        return self.model.predict(data)

    def predict_path(self, images):

        data = np.ndarray(shape=(1,224,224,3), dtype=np.float32)

        for img_path in images:

            print("Predicting image -> ", img_path)

            im = Image.open(img_path)

            im = ImageOps.fit(im, (224,224))
            
            image_array = np.asarray(im)

            norm_im = image_array
            #norm_im = (image_array.astype(np.float32)/127.0) - 1

            data[0] = norm_im

            res = self.infer(data)
            #res = self.model.predict(data)

            print("Predicted -> ", res)
            #print("Sigmoid -> ", tf.nn.sigmoid(res))

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

if __name__ == "main":
    t = Trainer("ere", 15, "resnet50", "imagenet", "model.ks", gpu=True)
    
