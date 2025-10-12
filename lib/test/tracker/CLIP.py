import torch
import cv2
import os

from transformers import CLIPModel, CLIPProcessor


class CLIP_Analysis:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

        # # build tokenizer, text_encoder and image_encoder from OpenAI/CLIP-base
        # clip_model = CLIPModel.from_pretrained("../huggingface_models/clip-vit-base-patch16")
        # self.processor = CLIPProcessor.from_pretrained("../huggingface_models/clip-vit-base-patch16")

        # build CLIPModel and CLIPProcessor from OpenAI/CLIP-large
        clip_model = CLIPModel.from_pretrained("../huggingface_models/clip-vit-large-patch14-336")
        self.processor = CLIPProcessor.from_pretrained("../huggingface_models/clip-vit-large-patch14-336")

        # move to gpu
        self.clip_model = clip_model.cuda()

    def initialize_all(self, image, info: dict):
        self.text = info['init_nlp']

        inputs = self.processor(text=self.text, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.cuda() for k, v in inputs.items()}

        outputs = self.clip_model(**inputs)
        logits_per_text = outputs.logits_per_text # this is the text-image similarity score

        return logits_per_text

    def track_all(self, image, info: dict = None):
        inputs = self.processor(text=self.text, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.cuda() for k, v in inputs.items()}

        outputs = self.clip_model(**inputs)
        logits_per_text = outputs.logits_per_text # this is the text-image similarity score

        return logits_per_text

    def initialize_once(self, image, info: dict):
        self.text = info['init_nlp']
        self.template = image

    def track_once(self, image, info: dict = None):
        self.search = image

        inputs = self.processor(text=self.text, images=[self.template, self.search], return_tensors="pt", padding=True)
        inputs = {k: v.cuda() for k, v in inputs.items()}

        outputs = self.clip_model(**inputs)
        logits_per_text = outputs.logits_per_text # this is the text-image similarity score
        probs = logits_per_text.softmax(dim=1) # we can take the softmax to get probabilities

        return logits_per_text, probs


def get_analyzer_class():
    return CLIP_Analysis
