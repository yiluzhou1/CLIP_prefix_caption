import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
import torchvision
from tqdm import tqdm
import argparse


def generate_json_file (clip_model_type: str, dataset_dir: str, input_text_name: str, out_dir: str, out_pkl_name: str):
    """
    The datest directory should have the following structure:
        dataset_dir/
            ├── input_text_name.txt (E.g. captions.txt)
            └── images/
    In each "images" folder, filenames of images are: "ROCO_00020.jpg", "ROCO_00027.jpg", etc...
    
    In input_text_name.txt (E.g. captions.txt), the content is stored as below:
    ROCO_00020	 Axial computed tomography scan of the pelvis showing a diffuse infiltration of the bladder wall, catheter in situ (arrow).
    ROCO_00027	 Postoperative anteroposterior radiograph of the pelvis.
    """
    image_dir = os.path.join(dataset_dir, "images")
    text_file_path = os.path.join(dataset_dir, f"{input_text_name}.txt")
    json_path = os.path.join(out_dir, f"{out_pkl_name}_{clip_model_type}" + ".json")

    # List to store the data objects
    data_list = []
    
    # Open the JSON file for writing
    with open(json_path, 'w') as json_file:
        # Read the captions.txt file
        with open(text_file_path, 'r') as file:
            for line in file:
                # Split the line into the image ID and caption
                try:
                    image_id, caption = line.strip().split('\t')
                except:
                    continue

                # Processing the caption content
                caption = caption.lower().rstrip().replace("\\n", "").rstrip(".")
                try:
                    # caption = caption.encode('ascii')
                    caption = caption.encode('ascii').decode('ascii')
                except:
                    continue
                # if len(caption) < 10: #Skip if the caption is too short
                #     continue

                # Construct the path to the image file
                image_path = os.path.join(image_dir, f'{image_id}.jpg')

                # Check if the image file exists
                if not os.path.exists(image_path):
                    continue
                
                # to make sure the file is a valid image
                try:
                    temp_data = torchvision.io.image.read_file(image_path)
                except:
                    print(image_path)
                    continue

                # Create the data object and add it to the list
                data = {
                    "image_path": image_path,
                    "caption": caption,
                    "image_id": image_id,
                }
                data_list.append(data)

        # Write the data to the JSON file
        json_file.write(json.dumps(data_list))
    
    return json_path


# for parsing roco dataset

def main(clip_model_type: str, dataset_dir: str, input_text_name: str, out_dir: str, out_pkl_name: str):
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"The folder '{out_dir}' has been created.")
    
    device = torch.device('cuda:0')
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    
    clip_model_type = clip_model_type.replace('/', '_')
    out_pkl_name = out_pkl_name.replace('.pkl', '')
    input_text_name = input_text_name.replace('.txt', '')
    json_path = generate_json_file (clip_model_type, dataset_dir, input_text_name, out_dir, out_pkl_name)
    
    out_pkl_path = os.path.join(out_dir, f"{out_pkl_name}_{clip_model_type}.pkl")
    with open(json_path, 'r') as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(len(data))):
        d = data[i]
        filename = d["image_path"]
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        d["clip_embedding"] = i
        all_embeddings.append(prefix)
        all_captions.append(d)
        if (i + 1) % 10000 == 0:
            with open(out_pkl_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_pkl_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    """
    Example: 
    python parse_roco.py --clip_model_type "ViT-B/32" --dataset_dir "/mnt/eds_data/gitrepos/roco-dataset/data/train/radiology" --input_text_name "captions.txt" --out_dir "./data/roco" --out_pkl_name "train"
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32")
    # choices=('RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px')
    parser.add_argument('--dataset_dir', default="/mnt/eds_data/gitrepos/roco-dataset/data/train/radiology")
    parser.add_argument('--input_text_name', default="captions.txt") 
    parser.add_argument('--out_dir', default="./data/roco")
    parser.add_argument('--out_pkl_name', default="train") # No need to add .pkl

    args = parser.parse_args()
    exit(main(args.clip_model_type, args.dataset_dir, args.input_text_name, args.out_dir, args.out_pkl_name))