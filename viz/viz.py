import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from lime import lime_image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


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


def plot(model, dataset, index):
    image, class_index = dataset[index]
    input_tensor = image.unsqueeze(0)

    rgb_img = tensor_to_rgb_image(image)

    # Target layer for CAM
    # target_layers = [model.vgg.vgg[8]]
    # target_layers = [model.inception.stage4[0]]
    target_layers = [model.stage1.downsample]
    target = [ClassifierOutputTarget(class_index)]

    lime_img = lime_explanation(image, model, class_index)
    cams = {
        "Grad-CAM": GradCAM(model=model, target_layers=target_layers),
        "Grad-CAM++": GradCAMPlusPlus(model=model, target_layers=target_layers),
        "Score-CAM": ScoreCAM(model=model, target_layers=target_layers),
    }

    cam_results = {"Original": rgb_img}
    for name, cam_method in cams.items():
        grayscale_cam = cam_method(input_tensor=input_tensor, targets=target)[0]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_results[name] = cam_image

    cam_results["LIME"] = lime_img

    plt.figure(figsize=(16, 8))
    for idx, (name, img) in enumerate(cam_results.items()):
        plt.subplot(2, 3, idx + 1)
        plt.imshow(img)
        plt.title(name)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
