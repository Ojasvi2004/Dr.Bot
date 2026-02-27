import torch
import torch.nn as nn
from torchvision.datasets import DatasetFolder
from torchvision.datasets import ImageFolder
from torch import cuda
from torchvision import datasets,transforms
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset,random_split


class MySimpleCNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
    
        self.features=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=32,out_channels=16,stride=1,kernel_size=3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=16,out_channels=8,stride=1,kernel_size=3,padding=1),
            nn.BatchNorm2d(8)
,            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )
        
        self.flattening=nn.Sequential(
            nn.Flatten()
        )
        self.flatten_dim=self.get_flatten_dim()
        
        self.classifier=nn.Sequential(
            nn.Linear(in_features=self.flatten_dim,out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32,out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16,out_features=classes),
   
        )
    
    def get_flatten_dim(self):
        with torch.no_grad():
            dummy=torch.zeros(1,1,64,64)
            dummy=self.features(dummy)
            return dummy.view(1,-1).size(1)
    
    def forward(self,x):
        x=self.features(x)
        x=self.flattening(x)
        x=self.classifier(x)
        return x

MyTranform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64,64)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5])]
)


Mydataset=datasets.ImageFolder(
    root="E:/DataSets/CuraLink/ImageClassifier",
    transform=MyTranform
)

train_size=(int)(0.8*len(Mydataset))
test_size=(int)(len(Mydataset)-train_size)

train_split,test_split=random_split(Mydataset,[train_size,test_size])

train_loader=DataLoader(
    dataset=train_split,
    shuffle=True,
    batch_size=64
)


test_loader=DataLoader(
    dataset=test_split,
    batch_size=64
)
device=torch.device("cuda" if torch.cuda.is_available()  else "cpu")
model=MySimpleCNN(classes=7)
model.to(device)

loss_function=nn.CrossEntropyLoss()
optimizer=optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4
)



def training_one_epoch():
    model.train()
    running_loss=0
    correct=0
    batch=0
    total=0
    for images,labels in train_loader:
        images=images.to(device)
        labels=labels.to(device)
        outputs=model(images)
        loss=loss_function(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss=running_loss+loss.item()
        preds=outputs.argmax(dim=1)
        correct=correct+(preds==labels).sum().item()
        batch=batch+1
        total=total+labels.size(0)
        print(f"""Total images processed {total} Loss in current batch {running_loss/batch}""")
    return running_loss/batch,correct/total

def validation():
    model.eval()
    running_loss=0
    correct=0
    batch=0
    total=0
    with torch.no_grad():
        for images,labels in test_loader:
            images=images.to(device)
            labels=labels.to(device)
            outputs=model(images)
            loss=loss_function(outputs,labels)
            running_loss=running_loss+loss.item()
            preds=outputs.argmax(dim=1)
            correct=correct+(preds==labels).sum().item()
            batch=batch+1
            total=total+labels.size(0)
        return running_loss/batch,correct/total
    
epochs=20
best_accuracy=0.0

def main():
    global best_accuracy
    global epochs
    for epoch in range(epochs):
        train_loss,train_accuracy=training_one_epoch()
        test_loss,test_accuracy=validation()
        print(f"""
        Epoch {epoch+1}/{epochs}
        Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f}
        Val   Loss: {test_loss:.4f} | Val   Acc: {test_accuracy:.4f}
        """)
        if(best_accuracy<test_accuracy):
            best_accuracy=test_accuracy
            torch.save(model.state_dict(),"imageClassifier.pth")
if __name__=="__main__":
    main()

    
        
    
    
    
    




        