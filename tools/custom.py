# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------

import glob
import argparse
import cv2
import os
import numpy as np
import _init_paths
import models
import torch
import torch.nn.functional as F
from PIL import Image

import tqdm
import time

import yaml


import torch.onnx
import onnx
import onnxruntime

mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
std = [0.225, 0.225, 0.225]

color_map = [(128, 64,128),
             (244, 35,232),
             ( 70, 70, 70),
             (102,102,156),
             (190,153,153),
             (153,153,153),
             (250,170, 30),
             (220,220,  0),
             (107,142, 35),
             (152,251,152),
             ( 70,130,180),
             (220, 20, 60),
             (255,  0,  0),
             (  0,  0,142),
             (  0,  0, 70),
             (  0, 60,100),
             (  0, 80,100),
             (  0,  0,230),
             (119, 11, 32)]

def parse_args():
    parser = argparse.ArgumentParser(description='Custom Input')
    
    parser.add_argument('--a', help='pidnet-s, pidnet-m or pidnet-l', default='pidnet-l', type=str)
    parser.add_argument('--c', help='cityscapes pretrained or not', type=bool, default=True)
    parser.add_argument('--p', help='dir for pretrained model', default='../pretrained_models/cityscapes/PIDNet_L_Cityscapes_test.pt', type=str)
    parser.add_argument('--r', help='root or dir for input images', default='../samples/', type=str)
    parser.add_argument('--t', help='the format of input images (.jpg, .png, ...)', default='.png', type=str)
    parser.add_argument('--f', help='path to the txt file that contains pathes of the directories to be converted', type=str )     

    args = parser.parse_args()

    return args

def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image

def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print('Attention!!!')
    print(msg)
    print('Over!!!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict = False)
    
    return model

def transform2torch(args):

    images_list = glob.glob(args.r+'*'+args.t)
    sv_path = args.r+'outputs/'

    model = models.pidnet.get_pred_model(args.a, 19 if args.c else 11)
    model = load_pretrained(model, args.p).cuda()
    model.eval()
    
    dummy_input = np.random.rand(370,1226,3) * 255
    dummy_input = dummy_input.transpose((2, 0, 1)).copy()
    dummy_input = torch.from_numpy(dummy_input).unsqueeze(0).cuda()       
    dummy_input = dummy_input.float()

    traced_script_module = torch.jit.trace(model, dummy_input)
    traced_script_module.save("PIDNet_s_cityscapes.pt")

def transform2oxnn(args):

    model = models.pidnet.get_pred_model(args.a, 19 if args.c else 11)
    model = load_pretrained(model, args.p).cuda()
    model.eval()

    with torch.no_grad():
        dummy_input = np.random.rand(370,1226,3) * 255
        dummy_input = dummy_input.transpose((2, 0, 1)).copy()
        dummy_input = torch.from_numpy(dummy_input).unsqueeze(0).cuda()       
        dummy_input = dummy_input.float()
        #dummy_input = torch.randn(1, 3, 224, 224).cuda()  
        input_names = [ "actual_input" ]
        output_names = [ "output_confidence" ]

        torch.onnx.export(model,
                        dummy_input,
                            "PIDNet_s_kitti_04.onnx",
                            verbose=False,
                            input_names=input_names,
                            output_names=output_names,
                            #dynamic_axes=dynamic_axes_dict,
                            opset_version=11,
                            export_params=True,
                            )

def testonnx(args):
    images_list = glob.glob(args.r+'*'+args.t)
    sv_path = args.r+'outputs/'


    ort_session = onnxruntime.InferenceSession("PIDNet_s_kitti_04.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    h = 0
    for img_path in images_list:
        img_name = img_path.split("\\")[-1]
        img_id = os.path.split(img_name)[1]
        img = cv2.imread(os.path.join(args.r, img_name),
                        cv2.IMREAD_COLOR)
        sv_img = np.zeros_like(img).astype(np.uint8)
        img = input_transform(img)

        img = img.transpose((2, 0, 1)).copy()
        img = torch.from_numpy(img).unsqueeze(0).cuda()

        # compute ONNX Runtime output prediction
        
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
        pred = ort_session.run(None, ort_inputs)[0]

        pred = np.argmax(pred, axis=1).squeeze(0)
        
        for i, color in enumerate(color_map):
            for j in range(3):
                sv_img[:,:,j][pred==i] = color_map[i][j]
        sv_img = Image.fromarray(sv_img)
        
        if not os.path.exists(sv_path):
            os.mkdir(sv_path)
        sv_img.save(sv_path+img_id)
        h+=1

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def segment_folder_onnx(args):

    file = open(args.f, 'r')
    directories  = file.read().splitlines()

    print(directories)

    ort_session = onnxruntime.InferenceSession("PIDNet_s_kitti_04.onnx")

    for dir in directories:
        
        print(f"{dir} started")

        sv_path = dir+'/semantic_image_0/'
        if not os.path.exists(sv_path):
            os.mkdir(sv_path)

        image_path = dir+'/image_0/'
        images_list = glob.glob(image_path + '*.png')
       
        for img_path in tqdm.tqdm(images_list):
            img_name = img_path.split("/")[-1]

            img = cv2.imread( img_path, cv2.IMREAD_COLOR)

            # cv2.imshow("asd", img)
            # cv2.waitKey(0)

            img = input_transform(img)

            img = img.transpose((2, 0, 1)).copy()
            img = torch.from_numpy(img).unsqueeze(0).cuda()

            # compute ONNX Runtime output prediction
            
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
            pred = ort_session.run(None, ort_inputs)[0]

            pred = np.argmax(pred, axis=1).squeeze(0)
            pred = pred.astype(np.uint8)

            # cv2.imshow("asd", pred)
            # cv2.waitKey(0)

            cv2.imwrite(sv_path+img_name, pred)


def segment_folder_pytorch(args):

    file = open(args.f, 'r')
    directories  = file.read().splitlines()

    print(directories)

    model = models.pidnet.get_pred_model('pidnet-s', 19)
    model = load_pretrained(model, '/cluster/home/oilter/PIDNet/pretrained_models/kitti/PIDNet_S_KITTI.pt').cuda()
    model.eval()

    for dir in directories:
        
        print(f"{dir} started")

        sv_path = dir+'/semantic_image_0/'
        if not os.path.exists(sv_path):
            os.mkdir(sv_path)

        image_path = dir+'/image_0/'
        images_list = glob.glob(image_path + '*.png')
       
        for img_path in tqdm.tqdm(images_list):
            img_name = img_path.split("/")[-1]

            img = cv2.imread( img_path, cv2.IMREAD_COLOR)

            # cv2.imshow("asd", img)
            # cv2.waitKey(0)

            img = input_transform(img)

            img = img.transpose((2, 0, 1)).copy()
            img = torch.from_numpy(img).unsqueeze(0).cuda()

            # compute output prediction
            
            pred = model(img)

            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
            pred = pred.astype(np.uint8)

            # cv2.imshow("asd", pred)
            # cv2.waitKey(0)

            cv2.imwrite(sv_path+img_name, pred)


def networkOut_folder_pytorch(args):

    file = open(args.f, 'r')
    directories  = file.read().splitlines()

    print(directories)

    model = models.pidnet.get_pred_model('pidnet-s', 19)
    model = load_pretrained(model, '/cluster/home/oilter/PIDNet/pretrained_models/kitti/PIDNet_S_KITTI.pt').cuda()
    model.eval()

    for dir in directories:
        
        print(f"{dir} started")

        sv_path = dir+'/semantic_file_0/'
        if not os.path.exists(sv_path):
            os.mkdir(sv_path)

        already_created_files = os.listdir(sv_path)


        image_path = dir+'/image_0/'
        images_list = glob.glob(image_path + '*.png')
       
        for img_path in tqdm.tqdm(images_list):
            img_name = img_path.split("/")[-1]

            if( sv_path+img_name+".yaml" in already_created_files):
                continue

            img = cv2.imread( img_path, cv2.IMREAD_COLOR)

            # cv2.imshow("asd", img)
            # cv2.waitKey(0)

            img = input_transform(img)

            img = img.transpose((2, 0, 1)).copy()
            img = torch.from_numpy(img).unsqueeze(0).cuda()

            # compute output prediction
            
            pred = model(img)
            pred = pred.detach().cpu().numpy()

            # with open(sv_path+img_name+".yaml", 'w') as f:
            #     yaml.dump(pred.tolist(), f)

            s = cv2.FileStorage(sv_path+img_name+".yaml", cv2.FileStorage_WRITE)
            s.write('network_output', pred)
            s.release()
            #cv2.Save("sv_path+img_name", cv2.fromarray(a))


            # pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
            # pred = pred.astype(np.uint8)

            # cv2.imshow("asd", pred)
            # cv2.waitKey(0)

            #cv2.imwrite(sv_path+img_name, pred)


def testPID(args):
    
    images_list = glob.glob(args.r+'*'+args.t)
    sv_path = args.r+'outputs/'

    model = models.pidnet.get_pred_model(args.a, 19 if args.c else 11)
    model = load_pretrained(model, args.p).cuda()
    model.eval()
    h = 0
    with torch.no_grad():
        for img_path in images_list:
            
            if h % 10 == 0:
                print("Going strong: ", h)
            img_name = img_path.split("\\")[-1]

            img_id = os.path.split(img_name)[1]
            img = cv2.imread(os.path.join(args.r, img_name),
                               cv2.IMREAD_COLOR)
            sv_img = np.zeros_like(img).astype(np.uint8)
            img = input_transform(img)
            img = img.transpose((2, 0, 1)).copy()
            img = torch.from_numpy(img).unsqueeze(0).cuda()
            pred = model(img)

            # pred = F.interpolate(pred, size=img.size()[-2:], 
            #                      mode='bilinear', align_corners=True)

            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
            
            for i, color in enumerate(color_map):
                for j in range(3):
                    sv_img[:,:,j][pred==i] = color_map[i][j]
            sv_img = Image.fromarray(sv_img)
            
            if not os.path.exists(sv_path):
                os.mkdir(sv_path)
            sv_img.save(sv_path+img_id)
            h += 1

def testPID_prob(args):
    images_list = glob.glob(args.r+'*'+args.t)
    sv_path = args.r+'outputs/'

    model = models.pidnet.get_pred_model(args.a, 19 if args.c else 11)
    model = load_pretrained(model, args.p).cuda()
    model.eval()
    h = 0
    with torch.no_grad():
        for img_path in images_list:
            img_name = img_path.split("\\")[-1]
            img = cv2.imread(os.path.join(args.r, img_name),
                               cv2.IMREAD_COLOR)
            sv_img = np.zeros_like(img).astype(np.uint8)
            img = input_transform(img)
            img = img.transpose((2, 0, 1)).copy()
            img = torch.from_numpy(img).unsqueeze(0).cuda()
            pred = model(img)

            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
            
            for i, color in enumerate(color_map):
                for j in range(3):
                    sv_img[:,:,j][pred==i] = color_map[i][j]
            sv_img = Image.fromarray(sv_img)
            
            if not os.path.exists(sv_path):
                os.mkdir(sv_path)
            sv_img.save(sv_path+str(h)+'.png')
            h += 1

if __name__ == '__main__':
    
    
    print("started")
    args = parse_args()

    networkOut_folder_pytorch(args)

    #segment_folder_pytorch(args)

    #segment_folder_onnx(args)

    #transform2torch(args)
    #transform2oxnn(args)
    #testonnx(args)
    #testPID(args)
    #testPID_prob(args)
