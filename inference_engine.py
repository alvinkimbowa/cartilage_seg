import os
import torch
import numpy as np
import cv2
from glob import glob
from archs import MonogenicUNeXt
from skimage import morphology, measure
from tqdm import tqdm
from PIL import Image
from scipy import ndimage

def load_model(model_path=None, img_size=(256, 256), device=None):
    # Load the model
    net_kwargs = {'input_channels': 0}
    net_kwargs = {
        "num_classes": 1,
        "input_channels": 1,
        "deep_supervision": False,
        "img_size": img_size[0],
        "input_channels": 0
    }
    monogenic_kwargs = {
        "img_size": img_size,
        "nscale": 8,
        "return_phase_asym": True,
        "return_phase": True,
    }   # use default values for now
    model = MonogenicUNeXt(monogenic_kwargs, net_kwargs)
    if model_path is None:
        model_path = "models/mono2d_unext_train_D1_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device), weights_only=False))
    model.eval()
    return model

def main(img_path_or_dir, save_dir=None, img_size=(256, 256), depth=30):
    '''
    Main function to perform inference on images.
    Args:
        img_path_or_dir: path to the image or directory containing images
        img_size: size of the image to be used for inference
        depth: depth of the cartilage in mm
    Returns:
        None
        Saves the predicted cartilage thickness for each image in the same directory.
    '''
    if os.path.isdir(img_path_or_dir):
        img_list = glob(f'{img_path_or_dir}/*.png')
    elif os.path.isfile(img_path_or_dir):
        img_list = [img_path_or_dir]
    else:
        raise ValueError(f"Invalid path: {img_path_or_dir}")
    
    if len(img_list) == 0:
        raise ValueError(f"No images found in {img_path_or_dir}")
    
    print(f"Found {len(img_list)} images in {img_path_or_dir}")
    print(f"Saving predictions to {save_dir}...")
    
    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(img_size=img_size, device=device)
    
    # Make predictions for each image
    for img_path in tqdm(img_list):
        # Load and process the image
        img = Image.open(img_path).convert('L')
        orig_size = img.size
        img = img.resize(img_size, Image.BILINEAR)
        img = np.array(img) / 255.0
        img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
        img.to(device)
        with torch.no_grad():
            pred = model(img)
        pred = pred.squeeze().cpu().numpy()
        pred = (pred > 0.5).astype(np.uint8)

        # 4. Find Largest Connected Component
        labeled, _ = ndimage.label(pred)
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  # Remove the background size (size[0] corresponds to background)
        largest_component = np.argmax(sizes)
        largest_component_mask = (labeled == largest_component).astype(np.uint8)

        # 5. Compute Cartilage Thickness (Only for the Largest Component)
        thickness = np.sum(largest_component_mask) / measure.perimeter(morphology.thin(largest_component_mask))
        thickness = thickness * (depth / img_size[0])  # Adjust thickness based on the original image size

        # Save the predicted cartilage mask
        if save_dir is None:
            save_dir = os.path.dirname(img_path)
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, os.path.basename(img_path.replace('.png', '_pred.png')))
        largest_component_mask = Image.fromarray((largest_component_mask * 255).astype(np.uint8))
        # largest_component_mask = largest_component_mask.resize(orig_size, Image.NEAREST)
        largest_component_mask.save(output_path)

        # 6. Overlay the mask on the original image       
        # Convert grayscale to BGR for color overlay
        img_color = cv2.cvtColor(img.squeeze().cpu().numpy() * 255, cv2.COLOR_GRAY2BGR)

        # Create a color mask for the largest connected component
        color_mask = np.zeros_like(img_color)
        color_mask[np.array(largest_component_mask) == 255] = [100, 255, 255]  # Yellow color for the mask

        # Overlay the mask onto the image
        alpha = 0.5  # Adjust transparency here
        image_out = cv2.addWeighted(img_color, 1, color_mask, alpha, 0)

        # Draw thickness value on the image
        thickness_text = f'Thickness: {thickness:.2f} mm'
        cv2.putText(image_out, thickness_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Save the overlay image
        overlay_path = os.path.join(save_dir, os.path.basename(img_path.replace('.png', '_pred_overlay.png')))
        cv2.imwrite(overlay_path, image_out.astype(np.uint8))
    print("Done!")

if __name__ == "__main__":
    depth = 30  # Imaging depth in mm
    img_size = (512, 512)
    img_path_or_dir = "sample_data/test_imgs"
    save_dir = "sample_data/test_ai_preds"
    main(img_path_or_dir, save_dir, img_size=img_size, depth=depth)
    