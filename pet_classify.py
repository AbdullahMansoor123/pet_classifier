import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import PIL.Image as Image
import matplotlib.pyplot as plt

class_map = ['cat', 'dog']
num_classes = len(class_map)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # logits here
        return x


model = CNN()
# model = torch.load('pet_classify_0.pth')
# model.load_state_dict(torch.load('pet_classify_0.pth'))
model.eval()

image_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def image_classify(model, image_transforms, image_path, classes):
    image = Image.open(image_path)
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    image = image_transforms(image).float()
    image = image.unsqueeze(0)
    outputs = model(image)
    prediction = torch.argmax(outputs, dim=1)
    # print(classes[prediction.item()])
    # show image with prediction
    plt.title(f'This is a {classes[prediction.item()]}')
    plt.axis('off')
    plt.show()

#Image Classifier


image_classify(model, image_transforms, 'cat1.jpg', class_map)

