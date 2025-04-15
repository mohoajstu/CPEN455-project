'''
This code is used to evaluate the FID score of the generated images.
You should at least guarantee this code can run without any error on test set.
And whether this code can run is the most important factor for grading.

We provide the remaining code, you can't modify other code, all you should do are:
1. Modify the sample function to get the generated images from your model and 
   ensure the generated images are saved to the gen_data_dir (line 20-33).
2. Modify how you call your sample function (line 50-55).

REQUIREMENTS:
- You should save the generated images to the gen_data_dir, which is fixed as './samples'
- If you directly run this code, it should generate images and calculate the FID score.
  You should follow the same format as the demonstration: there should be 100 images
  across 4 classes (each class has 25 images).
'''

from pytorch_fid.fid_score import calculate_fid_given_paths
from utils import *
from model import *
from dataset import *
import os
import torch
import argparse


sample_op = lambda x: sample_from_discretized_mix_logistic(x, 5)

def sample(model, n, obs, sample_op, label=None):
    """
    Class-conditional sampling if label is not None:
      - 'n' is how many images to generate
      - 'obs' is (C, H, W)
      - 'sample_op' picks from the predicted distributions
      - 'label' is a single integer label (applied to all 'n' samples)
    """
    device = next(model.parameters()).device
    x = torch.zeros(n, *obs, device=device)  # shape: [n, C, H, W]
    
    label_vec = None
    if label is not None:
        label_vec = torch.full((n,), label, dtype=torch.long, device=device)

    
    with torch.no_grad():
        for row in range(obs[1]):      # H dimension
            for col in range(obs[2]):  # W dimension
                out = model(x, label=label_vec, sample=False)
                x[:, :, row, col] = sample_op(out)[:, :, row, col]
    return x

def my_sample(model, gen_data_dir, sample_batch_size=25, obs=(3,32,32), sample_op=sample_op):
    """
    For each label in 'my_bidict', generate 25 images, save them to 'gen_data_dir'.
    """
    device = next(model.parameters()).device
    for label_name in my_bidict:
        label_idx = my_bidict[label_name]  
        print(f"Generating {sample_batch_size} images for class: {label_name}")
        sample_t = sample(model, sample_batch_size, obs, sample_op, label=label_idx)
        sample_t = rescaling_inv(sample_t)  
        save_images(sample_t, os.path.join(gen_data_dir), label=label_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--ref_data_dir', type=str,
                        default="data/test", help='Location for the reference dataset')
    args = parser.parse_args()
    
    ref_data_dir = args.ref_data_dir
    gen_data_dir = os.path.join(os.path.dirname(__file__), "samples")
    BATCH_SIZE = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(gen_data_dir):
        os.makedirs(gen_data_dir)

    model = conditional_pixelcnn(
        nr_resnet=2,        
        nr_filters=80,      
        nr_logistic_mix=5,  
        input_channels=3,
        nr_classes=4,
        emb_dim=80          
    )
    model = model.to(device)
    model.load_state_dict(torch.load('models/conditional_pixelcnn.pth', map_location=device))
    model.eval()

    my_sample(model=model, gen_data_dir=gen_data_dir)

    paths = [gen_data_dir, ref_data_dir]
    print("#generated images: {:d}, #reference images: {:d}".format(
        len(os.listdir(gen_data_dir)), len(os.listdir(ref_data_dir))))

    try:
        fid_score = calculate_fid_given_paths(paths, BATCH_SIZE, device, dims=192)
        print("Dimension {:d} works! fid score: {}".format(192, fid_score))
    except:
        print("Dimension {:d} fails!".format(192))
        fid_score = None

    print("Average fid score: {}".format(fid_score))
