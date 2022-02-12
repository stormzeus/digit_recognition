import torch, torchvision
from torch import nn
from torch import optim
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
from PIL import Image
from io import BytesIO
import requests
import sys
from board import mainloop
import pygame as pg



BATCH=64

T=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_data=torchvision.datasets.MNIST('mnist_data',train=True, download=True,transform=T)
val_data=torchvision.datasets.MNIST('mnist_data',train=False, download=False,transform=T)

train_dl=torch.utils.data.DataLoader(train_data,batch_size=BATCH)
val_dl=torch.utils.data.DataLoader(val_data,batch_size=BATCH)




def create_lenet():
    model=nn.Sequential(
        nn.Conv2d(1,6,5,padding=2),
        nn.ReLU(),
        nn.AvgPool2d(2,stride=2),
        

        nn.Conv2d(6,16,5,padding=0),
        nn.ReLU(),
        nn.AvgPool2d(2,stride=2),

        nn.Flatten(),
        nn.Linear(400,120),
        nn.ReLU(),
        nn.Linear(120,84),
        nn.ReLU(),
        nn.Linear(84,10)
    )
    return model

def validate(model,data):
    total,correct=0,0

    for i, (images,labels) in enumerate(data):
        images=images.cuda()
        x=model(images)
        value,pred=torch.max(x,1)
        pred=pred.data.cpu()
        total+=x.size(0)
        correct+=torch.sum(pred==labels)
    return correct*100./total

def train(epochs=3,lr=1e-3,device="cpu"):
    accuracies=[]
    cnn=create_lenet().to(device)
    cec=nn.CrossEntropyLoss()
    optimizer=optim.Adam(cnn.parameters(),lr=lr)
    max_accuracy=0

    for epoch in range(epochs):
        for i, (images,labels) in enumerate(train_dl):
            images=images.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            pred=cnn(images)
            loss = cec(pred,labels)
            loss.backward()
            optimizer.step()

        accuracy=float(validate(cnn,val_dl))
        accuracies.append(accuracy)

        print("Epoch:",epoch+1,"Accuracy",accuracy,"%")
        if accuracy>max_accuracy:
            best_model=copy.deepcopy(cnn)
            max_accuracy=accuracy
            print("saving best model with accuracy",accuracy)

        

    plt.plot(accuracies)
    plt.show()
    return best_model


        
if torch.cuda.is_available():
    device=torch.device('cuda:0')
else:
    device=torch.device('cpu')



# lenet=train(10,device=device)

# torch.save(lenet.state_dict(),"./digit_mnist.pt")

lenet=create_lenet().to(device)
lenet.load_state_dict(torch.load('./digit_mnist.pt'))


def predict_dl(model,data):
    y_pred,y_true=[],[]
    for i,(images,labels) in enumerate(data):
        images=images.cuda()
        x=model(images)
        value,pred=torch.max(x,1)
        pred=pred.data.cpu()
        y_pred.extend(list(pred.numpy()))
        y_true.extend(list(labels.numpy()))

    return np.array(y_pred), np.array(y_true)




def inference(path,model,device):
    # try:
    #     r=requests.get(path)
    # except:
    #     print("error getting image")
    #     sys.exit()
    
    img=Image.open(path).convert(mode='L')
    img=img.resize((28,28))
    x=(255-np.expand_dims(np.array(img),-1))/255.

    plt.imshow(x.squeeze(-1))
    plt.show()

    with torch.no_grad():
        pred=model(torch.unsqueeze(T(x),axis=0).float().to(device))
        return F.softmax(pred,dim=-1).cpu().numpy()

mainloop()
path="./image.png"


pred=inference(path,lenet,device=device)
pred_idx=np.argmax(pred)
print(f'predicted {pred_idx}, prob {pred[0][pred_idx]} %')
