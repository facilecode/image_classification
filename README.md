# Train/test image classifiers with MobileNet/ResNet

Each framework have its *_train.py script

    - keras_train.py
    - pytorch_train.py
    - tf_train.py

Execute with arguments as follows : script dataset epoch model_name

''' sh
python3 keras_train.py dataset/body_parts 15 body_parts
'''

or script framework dataset epoch model_name

''' sh
python3 train.py keras dataset/body_parts 15 body_parts
'''


To test execute with arguments as follows : script model_name framework gpu

With GPU acceleration
''' sh
python3 test.py body_parts pytorch gpu 
'''

Only CPU 

''' sh
python3 test.py body_parts pytorch  
'''

# From GUI this format should be received

{
    dataset_path: ...
    
}