import math
import os
import pathlib
import platform
import sys
from pathlib import Path

import cv2
from torchvision.transforms import ToTensor, CenterCrop
plt = platform.system()
if plt == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parent  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device, smart_inference_mode

IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation


class Bone_Age:
    def __init__(self, weights_path, model_name):
        self.device = torch.device("cpu")
        self.data = None
        self.half = False
        self.dnn = False
        model_dict = zip(model_name, weights_path)
        self.models = {key: DetectMultiBackend(v, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half) for
                       key, v in model_dict}

    @staticmethod
    def letterbox(im, new_shape=(224, 224), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im

    @staticmethod
    def __hist_img(im):
        r, g, b = im[..., :1], im[..., 1:2], im[..., 2:]
        clahe = cv2.createCLAHE(2.0, tileGridSize=(5, 5))
        equalized_r = clahe.apply(r)
        equalized_g = clahe.apply(g)
        equalized_b = clahe.apply(b)
        im = cv2.merge((equalized_r, equalized_g, equalized_b))
        cv2.imshow("im", im)
        return im

    @staticmethod
    def __calcBoneAge(score, sex):
        # 根据总分计算对应的年龄
        if sex == 'boy':
            boneAge = 2.01790023656577 + (-0.0931820870747269) * score + math.pow(score, 2) * 0.00334709095418796 + \
                      math.pow(score, 3) * (-3.32988302362153E-05) + math.pow(score, 4) * 1.75712910819776E-07 + \
                      math.pow(score, 5) * (-5.59998691223273E-10) + math.pow(score, 6) * 1.1296711294933E-12 + \
                      math.pow(score, 7) * (-1.45218037113138e-15) + math.pow(score, 8) * 1.15333377080353e-18 + \
                      math.pow(score, 9) * (-5.15887481551927e-22) + math.pow(score, 10) * 9.94098428102335e-26
            return round(boneAge, 2)
        elif sex == 'girl':
            boneAge = 5.81191794824917 + (-0.271546561737745) * score + \
                      math.pow(score, 2) * 0.00526301486340724 + math.pow(score, 3) * (-4.37797717401925E-05) + \
                      math.pow(score, 4) * 2.0858722025667E-07 + math.pow(score, 5) * (-6.21879866563429E-10) + \
                      math.pow(score, 6) * 1.19909931745368E-12 + math.pow(score, 7) * (-1.49462900826936E-15) + \
                      math.pow(score, 8) * 1.162435538672E-18 + math.pow(score, 9) * (-5.12713017846218E-22) + \
                      math.pow(score, 10) * 9.78989966891478E-26
            return round(boneAge, 2)

    @smart_inference_mode()
    def run(self,
            im: np.ndarray,
            weights_path="bone_age.pt",
            imgsz=(224, 224),
            conf_thres=0.6,
            iou_thres=0.25,
            max_det=1000,
            sex="girl",
            ):
        im = self.__hist_img(im)
        im0s = im.copy()
        model = DetectMultiBackend(weights_path, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        stride, names = model.stride, model.names
        im = self.letterbox(im, (224, 224), stride=stride)
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)
        imgsz = check_img_size(imgsz, s=stride)

        model.warmup(imgsz=(1, 3, *imgsz))
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)

        det = pred[0].detach().cpu().numpy()
        det = det[det[:, -1].argsort()].round()
        indices = np.unique(det[:, -1], return_index=True)[1]
        predict_list = np.unique(det[:, -1], return_counts=True)[1].tolist()
        if len(indices) != 7 or predict_list != [1, 1, 1, 4, 5, 4, 5]:
            print(predict_list)
            return "推理失败！", False
        Radius, Ulna, MCPF, MCP, PIP, MIP, DIP = np.split(det[:, :-2], indices[1:])
        if Radius[0][0] < Ulna[0][0]:
            Hand = False
        else:
            Hand = True
        Radius, Ulna, MCPF, MCP, PIP, MIP, DIP = [sorted(item, key=lambda x: x[[0]], reverse=Hand) for item in
                                                  [Radius, Ulna, MCPF, MCP, PIP, MIP, DIP]]
        arthrosis = {'MCPFirst': ['MCPFirst', MCPF[0]],
                     'MCPThird': ['MCP', MCP[1]],
                     'MCPFifth': ['MCP', MCP[3]],

                     'DIPFirst': ['DIPFirst', DIP[0]],
                     'DIPThird': ['DIP', DIP[2]],
                     'DIPFifth': ['DIP', DIP[4]],

                     'PIPFirst': ['PIPFirst', PIP[0]],
                     'PIPThird': ['PIP', PIP[2]],
                     'PIPFifth': ['PIP', PIP[4]],

                     'MIPThird': ['MIP', MIP[1]],
                     'MIPFifth': ['MIP', MIP[3]],

                     'Radius': ['Radius', Radius[0]],
                     'Ulna': ['Ulna', Ulna[0]], }
        SCORE = {'girl': {
            'Radius': [10, 15, 22, 25, 40, 59, 91, 125, 138, 178, 192, 199, 203, 210],
            'Ulna': [27, 31, 36, 50, 73, 95, 120, 157, 168, 176, 182, 189],
            'MCPFirst': [5, 7, 10, 16, 23, 28, 34, 41, 47, 53, 66],
            'MCPThird': [3, 5, 6, 9, 14, 21, 32, 40, 47, 51],
            'MCPFifth': [4, 5, 7, 10, 15, 22, 33, 43, 47, 51],
            'PIPFirst': [6, 7, 8, 11, 17, 26, 32, 38, 45, 53, 60, 67],
            'PIPThird': [3, 5, 7, 9, 15, 20, 25, 29, 35, 41, 46, 51],
            'PIPFifth': [4, 5, 7, 11, 18, 21, 25, 29, 34, 40, 45, 50],
            'MIPThird': [4, 5, 7, 10, 16, 21, 25, 29, 35, 43, 46, 51],
            'MIPFifth': [3, 5, 7, 12, 19, 23, 27, 32, 35, 39, 43, 49],
            'DIPFirst': [5, 6, 8, 10, 20, 31, 38, 44, 45, 52, 67],
            'DIPThird': [3, 5, 7, 10, 16, 24, 30, 33, 36, 39, 49],
            'DIPFifth': [5, 6, 7, 11, 18, 25, 29, 33, 35, 39, 49]
        },
            'boy': {
                'Radius': [8, 11, 15, 18, 31, 46, 76, 118, 135, 171, 188, 197, 201, 209],
                'Ulna': [25, 30, 35, 43, 61, 80, 116, 157, 168, 180, 187, 194],
                'MCPFirst': [4, 5, 8, 16, 22, 26, 34, 39, 45, 52, 66],
                'MCPThird': [3, 4, 5, 8, 13, 19, 30, 38, 44, 51],
                'MCPFifth': [3, 4, 6, 9, 14, 19, 31, 41, 46, 50],
                'PIPFirst': [4, 5, 7, 11, 17, 23, 29, 36, 44, 52, 59, 66],
                'PIPThird': [3, 4, 5, 8, 14, 19, 23, 28, 34, 40, 45, 50],
                'PIPFifth': [3, 4, 6, 10, 16, 19, 24, 28, 33, 40, 44, 50],
                'MIPThird': [3, 4, 5, 9, 14, 18, 23, 28, 35, 42, 45, 50],
                'MIPFifth': [3, 4, 6, 11, 17, 21, 26, 31, 36, 40, 43, 49],
                'DIPFirst': [4, 5, 6, 9, 19, 28, 36, 43, 46, 51, 67],
                'DIPThird': [3, 4, 5, 9, 15, 23, 29, 33, 37, 40, 49],
                'DIPFifth': [3, 4, 6, 11, 17, 23, 29, 32, 36, 40, 49]
            }
        }
        score = SCORE[sex]
        res_score = {}
        results = {}
        for finger, (model_name, box) in arthrosis.items():
            class_model = self.models[model_name]
            box = scale_boxes(im.shape[2:], box, im0s.shape).round()
            im2class = im0s[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
            im2class = T.Compose([ToTensor(), CenterCrop(224), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])(
                im2class)
            level = self.classify(class_model, im2class)
            results[finger] = level
            res_score[finger] = score[finger][level - 1]
        sum_score = sum(res_score.values())
        boneAge = self.__calcBoneAge(sum_score, sex)
        # 规范报告
        report = """
        第一掌骨骺分级{}级，得{}分；第三掌骨骨骺分级{}级，得{}分；第五掌骨骨骺分级{}级，得{}分；
        第一近节指骨骨骺分级{}级，得{}分；第三近节指骨骨骺分级{}级，得{}分；第五近节指骨骨骺分级{}级，得{}分；
        第三中节指骨骨骺分级{}级，得{}分；第五中节指骨骨骺分级{}级，得{}分；
        第一远节指骨骨骺分级{}级，得{}分；第三远节指骨骨骺分级{}级，得{}分；第五远节指骨骨骺分级{}级，得{}分；
        尺骨分级{}级，得{}分；桡骨骨骺分级{}级，得{}分。

        RUS-CHN分级计分法，受检儿CHN总得分：{}分，骨龄约为{}岁。""".format(
            results['MCPFirst'], res_score['MCPFirst'],
            results['MCPThird'], res_score['MCPThird'],
            results['MCPFifth'], res_score['MCPFifth'],
            results['PIPFirst'], res_score['PIPFirst'],
            results['PIPThird'], res_score['PIPThird'],
            results['PIPFifth'], res_score['PIPFifth'],
            results['MIPThird'], res_score['MIPThird'],
            results['MIPFifth'], res_score['MIPFifth'],
            results['DIPFirst'], res_score['DIPFirst'],
            results['DIPThird'], res_score['DIPThird'],
            results['DIPFifth'], res_score['DIPFifth'],
            results['Ulna'], res_score['Ulna'],
            results['Radius'], res_score['Radius'], sum_score, boneAge)
        return report, True

    @smart_inference_mode()
    def classify(self,
                 model=None,
                 im=None,
                 imgsz=(224, 224),
                 ):
        stride, names = model.stride, model.names
        imgsz = check_img_size(imgsz, s=stride)
        im = im.half() if model.fp16 else im.float()
        im = im.to(model.device)
        if len(im.shape) == 3:
            im = im[None]

        model.warmup(imgsz=(1, 3, *imgsz))
        results = model(im)
        pred = F.softmax(results, dim=1)
        top1i = pred[0].argsort(0, descending=True)[0].tolist()
        return top1i
