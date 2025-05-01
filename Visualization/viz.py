import contextlib
import gc
import os
import sys
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from lime import lime_image
from PIL import Image
from pytorch_grad_cam import (
    GradCAM,
    GradCAMPlusPlus,
    ScoreCAM,  # Needed for isinstance check
)
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm


def tensor_to_rgb_image(tensor):
    img = tensor.clone().detach().cpu()
    img = img * 0.5 + 0.5  # reverse normalization
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    return img


def lime_explanation(image, model, class_index):
    # Preprocess the image for LIME (needs to be in [0,1] range)
    image = image.cpu().numpy().transpose(1, 2, 0)  # C H W -> H W C
    image = np.clip(image, 0, 1)

    # Use LIME with SLIC segmentation
    explainer = lime_image.LimeImageExplainer()

    def predict_fn(images):
        # Make sure the images are in the correct format (N, H, W, C)
        images_tensor = torch.tensor(images).permute(0, 3, 1, 2).float()
        outputs = model(images_tensor)
        return F.softmax(outputs, dim=1).detach().cpu().numpy()

    # Get explanation from LIME
    explanation = explainer.explain_instance(image, predict_fn, top_labels=5, hide_color=0, num_samples=1000)

    # Get the image of the explanation for the target class index
    temp, mask = explanation.get_image_and_mask(class_index, positive_only=True, num_features=10, hide_rest=False)
    return temp


def plot_old(model, image, image_path, class_index, target_layers, output_dir="data/cam_outputs", verbose=False):
    # image, class_index = dataset[index]
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = image.unsqueeze(0)
    input_tensor = input_tensor.to(device)

    rgb_img = tensor_to_rgb_image(image)

    target = [ClassifierOutputTarget(class_index)]

    # lime_img = lime_explanation(image, model, class_index)
    # lime_img = lime_img.squeeze()  # remove dimensions like (1, 1, 3)
    # if lime_img.dtype != np.uint8:
    #     lime_img = (lime_img * 255).clip(0, 255).astype(np.uint8)  # scale to 0-255 if float

    # lime_path = os.path.join(output_dir, f"{base_name}_lime.jpg")
    # Image.fromarray(lime_img).save(lime_path)

    cams = {
        "Grad-CAM": GradCAM(model=model, target_layers=target_layers),
        "Grad-CAM++": GradCAMPlusPlus(model=model, target_layers=target_layers),
        "Score-CAM": ScoreCAM(model=model, target_layers=target_layers),
    }

    original_path = os.path.join(output_dir, f"{base_name}_original.jpg")
    Image.fromarray((rgb_img * 255).astype(np.uint8)).save(original_path)

    # cam_results = {"Original": rgb_img}
    for name, cam_method in cams.items():
        grayscale_cam = cam_method(input_tensor=input_tensor, targets=target)[0]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        # cam_results[name] = cam_image
        save_path = os.path.join(output_dir, f"{base_name}_{name}.jpg")
        Image.fromarray(cam_image).save(save_path)

    # cam_results["LIME"] = lime_img

    # if verbose:
    #     plt.figure(figsize=(16, 8))
    #     for idx, (name, img) in enumerate(cam_results.items()):
    #         plt.subplot(2, 3, idx + 1)
    #         plt.imshow(img)
    #         plt.title(name)
    #         plt.axis("off")
    #     plt.tight_layout()
    #     plt.show()


def plot(model, image, image_path, class_index, cams, output_dir="data/cam_outputs"):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    device = next(model.parameters()).device
    input_tensor = image.unsqueeze(0).to(device)
    rgb_img = tensor_to_rgb_image(image)

    target = [ClassifierOutputTarget(class_index)]

    # Save original image
    original_path = os.path.join(output_dir, f"{base_name}_original.jpg")
    Image.fromarray((rgb_img * 255).astype(np.uint8)).save(original_path)

    for name, cam_method in cams.items():
        if isinstance(cam_method, ScoreCAM):
            with torch.no_grad():
                grayscale_cam = cam_method(input_tensor=input_tensor, targets=target)[0]
        else:
            grayscale_cam = cam_method(input_tensor=input_tensor, targets=target)[0]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        save_path = os.path.join(output_dir, f"{base_name}_{name}.jpg")
        Image.fromarray(cam_image).save(save_path)
