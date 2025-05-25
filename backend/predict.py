import cv2
import numpy as np
import torch
from torch.nn import functional as F
from chars_dict import chars_dict, num_classes
from model import CRNN
import matplotlib.pyplot as plt

def load_model(path):
    model = CRNN(num_classes=num_classes)
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def greedy_decoder(output, chars_dict):
    # output: (T, B, C) — after log_softmax
    output = output.permute(1, 0, 2)  # (B, T, C)
    preds = torch.argmax(output, dim=2)  # (B, T)
    pred_strings = []

    for pred in preds:
        string = ""
        prev_char = None
        for idx in pred:
            char = idx.item()
            if char != prev_char and char != 0:
                string += chars_dict[char]
            prev_char = char
        pred_strings.append(string)

    return pred_strings

def predict(model, image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = img.shape

    img = cv2.resize(img, (int(58 / height * width), 58))

    # plt.figure(figsize=(15,2))
    # plt.imshow(img, cmap="gray")
    # plt.show()

    width = img.shape[1]
    img = np.pad(img, ((0, 0), (0, 1068 - width)), 'median')

    img = cv2.GaussianBlur(img, (5, 5), 0)

    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)

    img = np.expand_dims(img, axis=0)

    img = img / 255.

    img_tensor = torch.tensor(img, dtype=torch.float32)
    img_tensor = (img_tensor - 0.5) / 0.5  # normalize (mean=0.5, std=0.5)
    img_tensor = img_tensor.unsqueeze(0)  # (1, 1, H, W)

    output = model(img_tensor)
    log_probs = F.log_softmax(output, dim=2)
    prediction = greedy_decoder(log_probs, chars_dict)
    return prediction
