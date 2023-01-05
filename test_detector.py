from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import pickle as pkl
import itertools
import json

class test_net(nn.Module):
    def __init__(self, num_layers, input_size):
        super(test_net, self).__init__()
        self.num_layers= num_layers
        self.linear_1 = nn.Linear(input_size, 5)
        self.middle = nn.ModuleList([nn.Linear(5,5) for x in range(num_layers)])
        self.output = nn.Linear(5,2)
    
    def forward(self, x):
        x = x.view(-1)
        fwd = nn.Sequential(self.linear_1, *self.middle, self.output)
        return fwd(x)
        
def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    return img_



def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--scales", dest = "scales", help = "Scales to use for detection",
                        default = "1,2,3", type = str)
    
    return parser.parse_args()



class detector():
    def __init__(
        self, det='det', bs=1, confidence=0.5, nms_thresh=0.4, 
    cfgfile='cfg/yolov3.cfg', weightsfile='yolov3.weights', reso='416', scales='1,2,3'
    ) -> None:
        self.det = det
        self.batch_size = int(bs)
        self.confidence = float(confidence)
        self.nms_thesh = float(nms_thresh)
        self.start = 0
        self.scales = scales

        self.CUDA = torch.cuda.is_available()

        self.num_classes = 80
        self.classes = load_classes('data/coco.names') 

        #Set up the neural network
        print("Loading network.....")
        self.model = Darknet(cfgfile)
        self.model.load_weights(weightsfile)
        print("Network successfully loaded")
        
        self.model.net_info["height"] = reso
        self.inp_dim = int(self.model.net_info["height"])
        assert self.inp_dim % 32 == 0 
        assert self.inp_dim > 32

        #If there's a GPU availible, put the model on GPU
        if self.CUDA:
            self.model.cuda()
        
        
        #Set the model in evaluation mode
        self.model.eval()
        
        self.read_dir = time.time()
        
        
    def read_images(self, images='imgs') -> None:
        try:
            # ! changes
            # imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] =='.jpeg' or os.path.splitext(img)[1] =='.jpg']
            self.images = images
            self.imlist = [osp.join(images, img) for img in os.listdir(images) if os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] =='.jpeg' or os.path.splitext(img)[1] =='.jpg']
        except NotADirectoryError:
            self.imlist = []
            # imlist.append(osp.join(osp.realpath('.'), images))
            self.imlist.append(osp.join(images))
        except FileNotFoundError:
            print ("No file or directory with the name {}".format(images))
            exit()
        
        if not os.path.exists(self.det):
            os.makedirs(self.det)
            
            
    def detect_objects(self):
        self.load_batch = time.time()
    
        batches = list(map(prep_image, self.imlist, [self.inp_dim for x in range(len(self.imlist))]))
        self.im_batches = [x[0] for x in batches]
        self.orig_ims = [x[1] for x in batches]
        im_dim_list = [x[2] for x in batches]
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
        
        if self.CUDA:
            im_dim_list = im_dim_list.cuda()
        
        leftover = 0
        
        if (len(im_dim_list) % self.batch_size):
            leftover = 1
            
            
        if self.batch_size != 1:
            num_batches = len(self.imlist) // self.batch_size + leftover            
            self.im_batches = [torch.cat((self.im_batches[i*self.batch_size : min((i +  1)*self.batch_size,
                                len(self.im_batches))]))  for i in range(num_batches)]        

        i = 0
        
        write = False
        self.model(get_test_input(self.inp_dim, self.CUDA), self.CUDA)
        
        self.start_det_loop = time.time()
        
        objs = {}
        
        
        
        for batch in self.im_batches:
            #load the image 
            start = time.time()
            if self.CUDA:
                batch = batch.cuda()
            
            with torch.no_grad():
                prediction = self.model(Variable(batch), self.CUDA)
            
            
            prediction = write_results(prediction, self.confidence, self.num_classes, nms = True, nms_conf = self.nms_thesh)
            
            
            if type(prediction) == int:
                i += 1
                continue

            end = time.time()
            
                        
    #        print(end - start)

                

            prediction[:,0] += i*self.batch_size
            
        
                
            
            if not write:
                self.output = prediction
                write = 1
            else:
                self.output = torch.cat((self.output,prediction))
                
            
            

            for im_num, image in enumerate(self.imlist[i*self.batch_size: min((i +  1)*self.batch_size, len(self.imlist))]):
                im_id = i*self.batch_size + im_num
                objs = [self.classes[int(x[-1])] for x in self.output if int(x[0]) == im_id]
                print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/self.batch_size))
                print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
                print("----------------------------------------------------------")
            i += 1

            
            if self.CUDA:
                torch.cuda.synchronize()
        
        try:
            self.output
        except NameError:
            print("No detections were made")
            exit()
            
        im_dim_list = torch.index_select(im_dim_list, 0, self.output[:,0].long())
        
        scaling_factor = torch.min(self.inp_dim/im_dim_list,1)[0].view(-1,1)
        
        
        self.output[:,[1,3]] -= (self.inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
        self.output[:,[2,4]] -= (self.inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
        
        
        
        self.output[:,1:5] /= scaling_factor
        
        for i in range(self.output.shape[0]):
            self.output[i, [1,3]] = torch.clamp(self.output[i, [1,3]], 0.0, im_dim_list[i,0])
            self.output[i, [2,4]] = torch.clamp(self.output[i, [2,4]], 0.0, im_dim_list[i,1])
            
            
        self.output_recast = time.time()
        
        
        self.class_load = time.time()

        self.colors = pkl.load(open("pallete", "rb"))
        
        
        self.draw = time.time()
        
        
    # todo 这里可以修改框大小和字体
    def write(self, x, batches, results):
        c1 = tuple(map(int, x[1:3]))
        c2 = tuple(map(int, x[3:5]))
        img = results[int(x[0])]
        cls = int(x[-1])
        if not ((cls==0 or cls==2) and (c2[0]-c1[0])*(c2[1]-c1[1]) > 100):
            return
        label = "{0}".format(self.classes[cls])
        color = random.choice(self.colors)
        cv2.rectangle(img, c1, c2,color, 5)
        
        # output to file
        with open('./results/final_result{}.csv'.format(self.det[-2]), 'a+') as f:
                image_num = list(filter(lambda x: x.isdigit(), self.images))
                a = image_num[2]
                b = "".join(image_num[3:])
                np.savetxt(f, np.column_stack([c1[0], c1[1], c2[0], c2[1], cls, 0, int(a), int(b)]), fmt='%d', delimiter=',')
        
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
        
        

        
        
        return img
    
    
    def output_result(self):
        list(map(lambda x: self.write(x, self.im_batches, self.orig_ims), self.output))
        
        det_names = pd.Series(self.imlist).apply(lambda x: "{}/det_{}".format(self.det,x.split("/")[-1]))
        
        list(map(cv2.imwrite, det_names, self.orig_ims))
        
        end = time.time()
        
        # print()
        # print("SUMMARY")
        # print("----------------------------------------------------------")
        # print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
        # print()
        # print("{:25s}: {:2.3f}".format("Reading addresses", self.load_batch - self.read_dir))
        # print("{:25s}: {:2.3f}".format("Loading batch", self.start_det_loop - self.load_batch))
        # print("{:25s}: {:2.3f}".format("Detection (" + str(len(self.imlist)) +  " images)", self.output_recast - self.start_det_loop))
        # print("{:25s}: {:2.3f}".format("Output Processing", self.class_load - self.output_recast))
        # print("{:25s}: {:2.3f}".format("Drawing Boxes", end - self.draw))
        # print("{:25s}: {:2.3f}".format("Average time_per_img", (end - self.load_batch)/len(self.imlist)))
        # print("----------------------------------------------------------")

        
        torch.cuda.empty_cache()
        
    
        

    
