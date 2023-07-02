from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
from torchvision import models
import matplotlib.pyplot as plt
import os
from scipy import ndimage

from Model import Model
import cloudinary
import cloudinary.uploader



app = Flask(__name__)

cloudinary.config(
    cloud_name='dwo9yx1r8',
    api_key='731823634637836',
    api_secret='UUBWWpYlLL2YLBdT02DYI03mRmE'
)


MODEL_PATH = '/workspaces/ModelDeployment-Xray/model/model.pth'
model = Model()
model.load_state_dict(torch.load(MODEL_PATH),strict=False)
model.eval() 


loader = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
def image_loader(image_name):
     try:
        image = Image.open(image_name).convert("RGB")
        image = loader(image).float()
        image = image.unsqueeze(0)
        return image
     except FileNotFoundError:
        raise Exception("Image file not found.")
     except Exception as e:
        raise Exception(f"Error loading image: {str(e)}")
     


  

class LayerActivations():
    features = []

    def __init__(self, model):
        self.hooks = []
        self.hooks.append(model.model.layer4.register_forward_hook(self.hook_fn))

    def hook_fn(self, module, input, output):
        self.features.append(output)

    def remove(self):
        for hook in self.hooks:
            hook.remove()


def predict_img(path, model_ft):
    image_path = path
    img = image_loader(image_path)
    acts = LayerActivations(model_ft)
    device = next(model_ft.parameters()).device
    img = img.to(device)
    logps = model_ft(img)
    ps = torch.exp(logps)
    out_features = acts.features[0]
    acts.remove()  # Remove the hooks after extracting the feature maps
    out_features = torch.squeeze(out_features, dim=0)
    out_features = np.transpose(out_features.detach().cpu().numpy(), axes=(1, 2, 0))
    W = model_ft.model.fc[0].weight  # Access the weight property directly
    top_probs, top_classes = torch.topk(ps, k=2)
    pred = np.argmax(ps.detach().cpu())
    w = W[pred, :]
    cam = np.dot(out_features, w.detach().cpu())
    class_activation = ndimage.zoom(cam, zoom=(32, 32), order=1)
    img = img.cpu()
    img = torch.squeeze(img, 0)
    img = np.transpose(img, (1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    img = img.numpy()
    img = (img * std) + mean
    img = np.clip(img, a_max=1, a_min=0)
    return img, class_activation, pred



def predict_img1(image_path, model):
    # Replace this with your own image processing and prediction logic
    img = Image.open(image_path)
    class_activation, pred = predict_img(img,model)
    return img, class_activation, pred

def process_image1(img, class_activation):
    # Replace this with your own image processing logic
    processed_img = img.copy()
    # Process the image using the class activation map, etc.
    return processed_img

def save_image(img, path):
    # Convert NumPy array to PIL Image
    img_pil = Image.fromarray(np.uint8(img))
    # Save the PIL Image
    img_pil.save(path)



class_names = [0,1]
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400
    
    image = request.files['image']
    
    if image.filename == '':
        return jsonify({'error': 'No image selected.'}), 400
    
    # Save the uploaded image
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    image_path = os.path.join('uploads', image.filename)
    image.save(image_path)
    
    img, class_activation, pred = predict_img(image_path, model)
    pred_class = class_names[pred]
    
    # Process the image using the model (replace this with your own processing logic)
    processed_image = process_image1(img, class_activation)
    processed_image_path = os.path.join('uploads', 'processed_' + image_path.split('/')[-1])
    
    fig, axis = plt.subplots(1,2)
    axis[0].imshow(img,cmap="bone");
    axis[1].imshow(class_activation, cmap='jet', alpha=1);
    axis[1].imshow(processed_image,alpha=0.53);

    plt.tight_layout()
    plt.savefig(processed_image_path)

    try:
        # Upload the processed image to Cloudinary
        response = cloudinary.uploader.upload(processed_image_path)
        image_url = response['secure_url']

        # Delete the processed image file
        os.remove(processed_image_path)
        os.remove(image_path)

        # Create a response containing the predicted class and the processed image URL
        response = {
            'prediction': pred_class,
            'processed_image_url': image_url
        }

        return jsonify(response), 200

    except cloudinary.exceptions.Error as e:
        # Handle Cloudinary upload error
        error_message = str(e)
        return jsonify({'error': error_message}), 500

    except Exception as e:
        # Handle other exceptions
        error_message = str(e)
        return jsonify({'error': error_message}), 500



if __name__ == '__main__':
    app.run()
