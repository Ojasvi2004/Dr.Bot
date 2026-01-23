import cv2
import PIL
import torchvision 
import torch
from torchvision import transforms,datasets
import imageClassifier

# image=cv2.imread("cat.83.jpg")
image2=cv2.imread("D:/ML/test.jpg")

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=imageClassifier.MySimpleCNN(classes=7).to(device)
model.load_state_dict(torch.load("D:/ML/Dr.Bot/backend/app/imageclassifier/imageClassifier.pth", map_location=device,weights_only=True))
model.eval()

mytransform=transforms.Compose(
    [
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5])
    ]
)
mydataset=datasets.ImageFolder(
    root="E:/DataSets/CuraLink/ImageClassifier",
    transform=None
)

idx_to_class = {v: k for k, v in mydataset.class_to_idx.items()}
print(idx_to_class)

from PIL import Image

def predict_image(image):
    if image is None:
        raise FileNotFoundError("Could not load image")

    # OpenCV (BGR) → RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # NumPy → PIL
    image = Image.fromarray(image)

    # Apply SAME transform used in training
    image = mytransform(image)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred_idx = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][pred_idx].item()

    return idx_to_class[pred_idx], confidence

# print(predict_image(image))
print(predict_image(image2))
        