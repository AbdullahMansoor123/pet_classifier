import torch
from cv2 import cv2
import numpy as np
from time import time
from cnn_models import CNN
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import PIL.Image as Image
import matplotlib.pyplot as plt

class_map = ['cat', 'dog']
num_classes = len(class_map)

class ObjectDetection:
    """
    the class uses a model trained on custom images to identify person
    """
    def __init__(self, url, class_map, out_file='custom_results.mp4'):
        """initializes the class with video_name and out_file"""
        self._URL = url
        self.class_map = class_map
        self.out_file = out_file
        self.model = self.load_model()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_video_from_url(self):
        return cv2.VideoCapture(self._URL)

    def load_model(self):
        """load saved trained model trained on custom data"""
        model = CNN()
        model = torch.load('pet_classify.pth')
        return model

    def score_frame(self,frame):
        self.model.to(self.device)
        frame =[frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        return labels, cord

    def class_to_label(self,x):
        """
        for a given label value, returns the corresponding string output
        """
        return self.class_map[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        player = self.get_video_from_url()
        # assert player.isOpened()
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        four_cc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(self.out_file, four_cc, 20, (x_shape, y_shape))
        while True:
            start_time = time()
            ret, frame = player.read()
            # assert ret
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time()
            fps = 1 / np.round(end_time - start_time, 3)
            print(f"Frames Per Second : {fps}")
            out.write(frame)


# Create a new object and execute.
a = ObjectDetection('cat_dog_video',class_map)
a()

#
# # Reading IP Camera
# filename = 'cat_dog_video.mp4'
# # noinspection PyArgumentList
# video = cv2.VideoCapture(0)
# while True:
#     ret, frame = video.read()
#     cv2.imshow('frame', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# video.release()
# cv2.destroyAllWindows()

#
# #load the model
#

#
#
# model = torch.load('pet_classify.pth')
# model.eval()
#
# image_transforms = transforms.Compose([
#     transforms.Resize(255),
#     transforms.CenterCrop(224),
#     transforms.RandomHorizontalFlip(0.5),
#     transforms.AutoAugment(),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])
# ])
# #Socoring a single frame
# def image_classify(model, image_transforms, image_path, classes):
#     image = Image.open(image_path)
#     plt.figure(figsize=(5, 5))
#     plt.imshow(image)
#     image = image_transforms(image).float()
#     image = image.unsqueeze(0)
#     outputs = model(image)
#     prediction = torch.argmax(outputs, dim=1)
#     # print(classes[prediction.item()])
#     # show image with prediction
#     plt.title(f'This is a {classes[prediction.item()]}')
#     plt.axis('off')
#     plt.show()
# #Socoring a single frame
#
# while True:
#     ret, frame = video.read() #ret store results of frame availability
#     image_classify(model, image_transforms, frame, class_map)
#
