import argparse
import torch

parser = argparse.ArgumentParser(description='dataset_path epochs model_name')

parser.add_argument("-framework", type=str, required=True)
parser.add_argument("-dataset", type=str, required=True)
parser.add_argument("-base", type=str, required=True)
parser.add_argument("-w", type=str, required=True)
parser.add_argument("-epochs", type=int, required=False, default=1)
parser.add_argument("-batch", type=int, required=False, default=4)
parser.add_argument("-gpu", action="store_true")

args = parser.parse_args()

def main():
    if args.framework == "torch":
        from core import pytorch_class
        m = pytorch_class.Trainer(dataset_path=args.dataset, base_model=args.base, weights=args.w, epochs=args.epochs, batch=args.batch, model_name="env", gpu=args.gpu)
        #m.train()
    if args.framework == "keras": 
        from core import keras_class
        m = keras_class.Trainer(dataset_path=args.dataset, base_model=args.base, weights=args.w, epochs=args.epochs, batch=args.batch, model_name="env", gpu=args.gpu)

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()

"""
python train.py -framework torch -dataset datasets/env/original -base mobilenetv2 -w imagenet -epochs 20 -gpu
python train.py -framework keras -dataset datasets/env/original -base mobilenetv2 -w imagenet -epochs 20
"""