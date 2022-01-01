#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 09:59:58 2021

@author: Pablo Navarrete Michelini
"""
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import torchvision.transforms.functional as TF

from models import edgeSR_MAX, edgeSR_TM, edgeSR_CNN, edgeSR_TR, FSRCNN, ESPCN, Classic


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [
        '.png', '.tif', '.jpg', '.jpeg', '.bmp', '.pgm', '.PNG'
    ])


if __name__ == '__main__':
    # ############################# CHANGE HERE ###############################
    device = torch.device('cuda:0')
    model_file = 'model-files/Bicubic_s2.model'
    # #########################################################################

    model_id = model_file.split('.')[-2].split('/')[-1]
    torch.backends.cudnn.benchmark = True
    with torch.cuda.device(device):
        with torch.no_grad(), torch.jit.optimized_execution(True):
            print('\n- Load model')
            if model_id.startswith('eSR-MAX_'):
                model = edgeSR_MAX(model_id)
            elif model_id.startswith('eSR-TM_'):
                model = edgeSR_TM(model_id)
            elif model_id.startswith('eSR-TR_'):
                model = edgeSR_TR(model_id)
            elif model_id.startswith('eSR-CNN_'):
                model = edgeSR_CNN(model_id)
            elif model_id.startswith('FSRCNN_'):
                model = FSRCNN(model_id)
            elif model_id.startswith('ESPCN_'):
                model = ESPCN(model_id)
            elif model_id.startswith('Bicubic_'):
                model = Classic(model_id)
            else:
                assert False
            model.load_state_dict(
                torch.load(model_file, map_location=lambda storage, loc: storage),
                strict=True
            )
            model.to(device).half()

            input_list = [
                str(f) for f in Path('input').iterdir() if is_image_file(f.name)
            ]
            for input_file in tqdm(input_list):
                input_tensor = TF.to_tensor(
                    Image.open(input_file).convert('L')
                ).unsqueeze(0).to(device).half()

                output_rgb = model(
                    input_tensor
                ).data[0].clamp(0, 1.).expand(3, -1, -1).permute(1, 2, 0).cpu().numpy() * 255.

                Image.fromarray(
                    np.uint8(np.round(output_rgb))
                ).save(
                    'output/' + '.'.join(input_file.split('/')[-1].split('.')[:-1]) + '_' + model_id + '.png'
                )
