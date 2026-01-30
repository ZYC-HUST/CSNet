import os
import cv2
import glob
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from CSNet import CSNet
from utils import *

def test(model, device, dataset_name, result_dir):
    model.eval()

    output_dir = os.path.join(result_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"----- Starting test on [{dataset_name}] dataset -----")
    print(f"Fusion results will be saved to: {output_dir}")

    image_pairs = []
    search_path_display = ""

    if dataset_name == 'Lytro':
        source_dir = "data/Lytro"
        file_a_suffix, file_b_suffix = "-A.jpg", "-B.jpg"
        search_path_display = os.path.join(source_dir, f'*{file_a_suffix}')
        A_files = sorted(glob.glob(search_path_display))
        for a_path in A_files:
            base_name = os.path.basename(a_path).replace(file_a_suffix, '')
            b_path = os.path.join(source_dir, f'{base_name}{file_b_suffix}')
            image_pairs.append((a_path, b_path, base_name))

    elif dataset_name == 'MFFW':
        source_dir = "data/MFFW"
        file_a_suffix, file_b_suffix = "_A.jpg", "_B.jpg"
        search_path_display = os.path.join(source_dir, f'*{file_a_suffix}')
        A_files = sorted(glob.glob(search_path_display))
        for a_path in A_files:
            base_name = os.path.basename(a_path).replace(file_a_suffix, '')
            b_path = os.path.join(source_dir, f'{base_name}{file_b_suffix}')
            image_pairs.append((a_path, b_path, base_name))

    elif dataset_name == 'MFI-WHU':
        source_1_dir = "data/MFI-WHU/source_1"
        source_2_dir = "data/MFI-WHU/source_2"
        search_path_display = os.path.join(source_1_dir, '*.jpg')
        A_files = sorted(glob.glob(search_path_display))
        for a_path in A_files:
            base_name_ext = os.path.basename(a_path)
            base_name = os.path.splitext(base_name_ext)[0]
            b_path = os.path.join(source_2_dir, base_name_ext)
            image_pairs.append((a_path, b_path, base_name))

    elif dataset_name == 'SIMIF':
        source_dir = "data/SIMIF"
        file_a_suffix, file_b_suffix = "left.jpg", "right.jpg"
        search_path_display = os.path.join(source_dir, f'*{file_a_suffix}')
        A_files = sorted(glob.glob(search_path_display))
        for a_path in A_files:
            base_name = os.path.basename(a_path).replace(file_a_suffix, '')
            b_path = os.path.join(source_dir, f'{base_name}{file_b_suffix}')
            image_pairs.append((a_path, b_path, base_name))
            
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    if not image_pairs:
        print(f"Error: No test images found at path '{search_path_display}'.")
        return

    results_list = []

    for A_file_path, B_file_path, base_name in image_pairs:
        if not os.path.exists(B_file_path):
            print(f"Warning: Skipping {base_name}, corresponding B image not found.")
            continue
        
        imgA_pil = Image.open(A_file_path).convert('RGB')
        imgB_pil = Image.open(B_file_path).convert('RGB')

        to_tensor = transforms.ToTensor()
        imgA = to_tensor(imgA_pil).unsqueeze(0).to(device)
        imgB = to_tensor(imgB_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_maskA, pred_maskB, pred_imageC = model(imgA, imgB)
            imageF = similarity_selection(pred_maskA, pred_maskB, pred_imageC, imgA, imgB, metric='perceptual')

        fused_np = (imageF.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        output_path = os.path.join(output_dir, f"{base_name}-Fused.png")
        cv2.imwrite(output_path, cv2.cvtColor(fused_np, cv2.COLOR_RGB2BGR))

        imgA_np = np.array(imgA_pil)
        imgB_np = np.array(imgB_pil)
        fused_gray = cv2.cvtColor(fused_np, cv2.COLOR_RGB2GRAY)
        A_gray = cv2.cvtColor(imgA_np, cv2.COLOR_RGB2GRAY)
        B_gray = cv2.cvtColor(imgB_np, cv2.COLOR_RGB2GRAY)

        current_results = {
            'Q_MI': metricNMI(A_gray, B_gray, fused_gray),
            'MI': metricMI_standard(A_gray, B_gray, fused_gray),
            'Q_NCIE': metricWang(A_gray, B_gray, fused_gray),
            'Q_Y': metricYang(A_gray, B_gray, fused_gray),
            'Q_G': metricXydeas(A_gray, B_gray, fused_gray),
        }
        results_list.append(current_results)

    if not results_list:
        print(f"Warning: No images were successfully processed for the {dataset_name} dataset.")
        return

    metrics_to_average = ['Q_MI', 'MI', 'Q_NCIE', 'Q_Y', 'Q_G']
    avg_results = {metric: np.mean([res[metric] for res in results_list]) for metric in metrics_to_average}
    
    print(f"----- Test completed on [{dataset_name}] dataset -----")
    print("Average performance metrics:")
    for metric, value in avg_results.items():
        print(f"  {metric}: {value:.4f}")
    print("-" * 40)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = 'checkpoint/checkpoint.pth'
    
    print("Initializing model...")
    model = CSNet(in_channels=3, base_channels=16, num_levels=4)
    model.to(device)

    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    state_dict = checkpoint['model_state_dict']
    if next(iter(state_dict)).startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] 
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    
    print("Model weights loaded successfully.")

    result_dir = "results"
    os.makedirs(result_dir, exist_ok=True)

    test(model, device, 'Lytro', result_dir)
    test(model, device, 'MFFW', result_dir)
    test(model, device, 'MFI-WHU', result_dir)
    test(model, device, 'SIMIF', result_dir)

    print("All tests have been completed.")

if __name__ == '__main__':
    main()