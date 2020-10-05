import torch
import torchvision
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models as models
import torchvision.transforms as trans
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from PIL import Image
import cv2
from time import sleep
import os
import copy
import time
import json
import numpy as np 
from glob import glob
from core import debugger 

class Trainer:

    train_transforms_augm = trans.Compose([
        #trans.Resize(224),
        #trans.RandomCrop(224),
        trans.RandomResizedCrop(224),
        trans.RandomHorizontalFlip(),
        #trans.ColorJitter(brightness=(0.10,0.9), contrast=(0.10,0.9)),
        trans.ToTensor(),
        trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_transforms = trans.Compose([
        trans.Resize(224),
        trans.ToTensor(),
        trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transforms = trans.Compose([
        trans.Resize((224,224)),
        trans.ToTensor(),
        trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # to-do add optimizer choice
    def __init__(self, dataset_path, base_model, weights, epochs, batch, model_name, gpu):

        self.model = None
        self.model_name = model_name
        self.classes = os.listdir(dataset_path)
        self.model_info = {
            "base_model": None,
            "weights": None,
            "model": None,
            "classes": self.classes
        }
        print("Found classes -> ", self.classes)
        self.init_model(base_model, weights, len(self.classes))
        self.train(dataset_path, epochs, batch, gpu)

    
    def init_model(self, base_model, weights, output_dim):

        pretrained = None
        if weights == "random":
            pretrained = False
        elif weights == "imagenet":
            pretrained = True

        if base_model == "mobilenetv2":
            self.model = models.mobilenet_v2(pretrained=pretrained)
            self.model.classifier = nn.Sequential(
                nn.Linear(
                    in_features = 1280, 
                    out_features = output_dim
                ), 
                nn.Sigmoid()
            )

        if "resnet" in base_model:
            if base_model == "resnet18":
                self.model = models.resnet18(pretrained=pretrained)
            if base_model == "resnet34":
                self.model = models.resnet34(pretrained=pretrained)
            if base_model == "resnet50":
                self.model = models.resnet50(pretrained=pretrained)

            self.model.fc = nn.Sequential(
                nn.Linear(
                    in_features = self.model.fc.in_features,
                    out_features = output_dim
                ),
                nn.Sigmoid()
            )
        
        self.model_info["base_model"] = base_model

        self.model.eval()

    def split_dataset(self, split):
        pass

    def get_train_val_loaders(self, dataset_path, batch, split_val):

        trainset = torchvision.datasets.ImageFolder(
            root=dataset_path,
            transform=self.train_transforms_augm
        )
        valset = torchvision.datasets.ImageFolder(
            root=dataset_path,
            transform=self.test_transforms
        )

        num_train = len(trainset)
        indices = list(range(num_train))
        split = int(np.floor(split_val * num_train))

        np.random.shuffle(indices)

        train_idx, val_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch, sampler=train_sampler
        )
        testloader = torch.utils.data.DataLoader(
            valset, batch_size=batch, sampler=val_sampler
        )

        return trainloader, testloader

    # train
    def train(self, dataset_path, epochs, batch, gpu):

        since = time.time()

        if gpu:
            self.model.cuda()

        transform = self.train_transforms_augm
        # to-do : separate dataset into train/val
        dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)

        print("dataset size ", len(dataset))
        full = len(dataset)
        train = int(0.8 * full)
        test = full - train

        trainset, valset = torch.utils.data.random_split(dataset, [train, test])

        print("Full -> ", len(dataset))
        print("Train -> ", len(trainset))
        print("Test -> ", len(valset))

        """
        trainset = torchvision.datasets.ImageFolder(root=dataset_path + '/train', transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=8)

        testset = torchvision.datasets.ImageFolder(root=dataset_path + '/val', transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=8)
        """
        
        #trainset, valset = self.get_train_val_loaders(dataset_path, 0.2)
    
        dataloaders_dict = {
            'train': torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=0),
            'val': torch.utils.data.DataLoader(valset, batch_size=batch, shuffle=False, num_workers=0),
        }

        dataset_sizes = {
            'train': len(trainset), 
            'val': len(valset)
        }

        print("Train : ", len(trainset))
        print("Val : ", len(valset))

        #to-do : optimizers ...
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                step=0
                for inputs, labels in dataloaders_dict[phase]:
                    print(f"Step {step}/{len(dataloaders_dict[phase])} -- {epoch}/{epochs} epochs")
                    step = step + 1

                    if gpu == True:
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        print("saving ...")

        self.model.load_state_dict(best_model_wts)

        torch.save(best_model_wts, "models/torch/" + self.model_name + ".pth")
        self.model_info["weights"] = "models/torch/" + self.model_name + ".pth"

        torch.save(self.model, "models/torch/full_" + self.model_name + ".pth")
        self.model_info["model"] = "models/torch/full_" + self.model_name + ".pth"

        with open("models/torch/" + self.model_name + ".json", "w") as f:
            json.dump(self.model_info, f)
            
    # predict on image from image_path
    def predict(self, image_path, camera=None):

        if camera == None:
            im = cv2.imread(image_path)
        else:
            ok, im = self.cam.read()

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im)

        im = self.test_transforms(im).unsqueeze(0)

        output = self.model(im)
        _, preds = torch.max(output, 1)

        print(preds)

# Test
class Tester:
    test_transforms = trans.Compose([
        trans.Resize((224,224)),
        trans.ToTensor(),
        trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # model_path is the models name with .json extension
    #   Exemple: model_path = "model.json" 
    #       
    #       - "base_model": mobilenetv2
    #       - "weights": pytorch_models/model.pth 
    #       - "classes" ["hand", "leg"]
    #
    def __init__(self, model_path, full):
        f = open(model_path)
        self.model_info = json.load(f)
        self.init_model(full)
        
    def init_model(self, full):
        
        if full:
            self.model = torch.load(self.model_info["model"], map_location=torch.device("cpu"))
            self.model.eval()
            print(self.model)
            return

        if self.model_info["base_model"] == "mobilenetv2":
            self.model = models.mobilenet_v2()
            self.model.classifier = nn.Sequential(
                nn.Linear(
                    in_features = 1280, 
                    out_features = len(self.model_info["classes"])
                ), 
                nn.Sigmoid()
            )

        if "resnet" in self.model_info["base_model"]:
            if self.model_info["base_model"] == "resnet18":
                self.model = models.resnet18()
            if self.model_info["base_model"] == "resnet34":
                self.model = models.resnet34()
            if self.model_info["base_model"] == "resnet50":
                self.model = models.resnet50()

            self.model.fc = nn.Sequential(
                nn.Linear(
                    in_features = self.model.fc.in_features,
                    out_features = len(self.model_info["classes"])
                ),
                nn.Sigmoid()
            )

        self.model.load_state_dict(torch.load(self.model_info["weights"], map_location=torch.device("cpu")))
        self.model.eval()

        print(self.model)
    
    @debugger.timeit
    def infer(self, data):
        return  self.model(data)

    def predict_path(self, images):

        for im_path in images:

            print("Predicting -> ", im_path)

            im = Image.open(im_path)

            im = self.test_transforms(im).unsqueeze(0)

            output = self.infer(im)
            #output = self.model(im)

            _, preds = torch.max(output, 1)

            print(output, preds)

            