import torch
import clip
import PIL
from PIL import Image
from torchvision.io import read_image

from config.core import device


class CLIP_classify:
    def __init__(self, backbone="ViT-B/32"):
        self.model, self.preprocess = clip.load(backbone, device=device)

    def classify(self, cls_names, image: PIL.Image):
        cls_names_tokenized = clip.tokenize(cls_names).to(device)
        image = self.preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(image, cls_names_tokenized)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        return cls_names[probs.argmax()]