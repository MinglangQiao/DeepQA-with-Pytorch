
from datasets.dataloader import LIVEDatasetLoader, TID2013DatasetLoader, TID2013_GTtable
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.misc as m

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models.deepQA import deepIQA_model as predictNet

import skimage.measure

from functools import partial
import pickle

### ml add
import cv2
from skimage.color import rgb2gray
from datasets.preprocessing import *
import scipy.misc as m
from PIL import Image
import torchvision.transforms as transforms

class test_momo():

    def __init__(self):
        
        self.base_dir =  "/home/ml/1_code_base/IQA/data/MOMO_dataset/"
        self.ref_image_path = self.base_dir + "test2_082.bmp"
        self.dis_image_path = self.base_dir + "vzzm_082.bmp"
        self.dis_momo_path = self.base_dir + "momo_082.bmp"
        self.img_size = (384, 512)
        self.x_offset = 100
        self.y_offset = 800

    def my_error_map(self, img1, img2):
        return (img1-img2) 

    def preprocessing_vvzm(self):
        
        ref_image = m.imread(self.ref_image_path)[self.y_offset:self.y_offset+self.img_size[0], 
            self.x_offset:self.x_offset+self.img_size[1]]
        dis_image = m.imread(self.dis_image_path)[self.y_offset:self.y_offset+self.img_size[0], 
            self.x_offset:self.x_offset+self.img_size[1]]
        # dis_momo_image = cv2.imread(self.dis_momo_path)

        # print(np.shape(ref_image), np.shape(dis_image))
        # t

        img_gray = rgb2gray(dis_image)
        img0_gray = rgb2gray(ref_image)
        # img_d = img_gray
        # img_r = img0_gray
        img_d = low_frequency_sub(img_gray*255)
        img_r = low_frequency_sub(img0_gray*255)
        # print(np.max(img_d), np.max(img_r)) ## (0.6005725490196079, 0.4989470588235296)
        # t

        # error = self.my_error_map(img_d, img_r)
        error = error_map(img_d, img_r, epsilon=1.)
        # print("ml: ", np.shape(img_d), np.shape(error))
        # t
        # plt.imshow(error, cmap="winter")
        # plt.show()

        img_d = Image.fromarray(img_d)
        img_d = transforms.ToTensor()(img_d)
        error = torch.from_numpy(error).float()
        # print("ml: ", np.shape(img_d), np.shape(error))
        # t
        return img_d, error, dis_image, ref_image

def showfigures_TID13():
    def __find_test_img(gt_score):
        gt_table, _, _ = TID2013_GTtable('../data/TID2013_dataset/mos_with_names.txt')

        for idx, sample in enumerate(gt_table):

            if gt_score == float(sample[1]):
                img_path = '../data/TID2013_dataset/distorted_images/%s' % (gt_table[idx, 0])
                ref_path = '../data/TID2013_dataset/reference_images/I%s.BMP' % (sample[0][1:3])

        img = m.imread(img_path)
        ref = m.imread(ref_path)

        return img, ref

    testset = TID2013DatasetLoader('../data/TID2013_dataset/',
                                   train_phase=False,
                                   is_sample=False,
                                   seed=12,
                                   global_permute=False)
    testloader = DataLoader(testset,
                            shuffle=False,
                            batch_size=1,
                            num_workers=1,
                            pin_memory=True)

    # load the trained model
    model = predictNet()
    ## ml add
    the_used_model = "snapshots/deepQA_TID13_seed32_0.9286_0.9244_epoch3780.pth"
    model_dict = torch.load(the_used_model)['model']
    model.load_state_dict(model_dict)  # copyWeights(model, model_dict, freeze=False)
    model.eval()
    model.to('cuda')

    test_momo1 = test_momo()

    for batch_id, (img, error, score_gt) in enumerate(testloader):

        # print(np.shape(img))
        # t
        print(">>>>>>>>>>>batch_id: ", batch_id, np.shape(img), np.shape(error)) # (1, 1, 384, 512), (1, 384, 512)
        # t
        img1, error1, img_ori, ref_ori = test_momo1.preprocessing_vvzm()
        img = img1[np.newaxis, :]
        error = error1[np.newaxis, :]
        # print(">>>>>>>>>>>batch_id: ", batch_id, np.shape(img2), np.shape(error2))
        # t
        score_gt = score_gt.type('torch.FloatTensor')
        img, error, score_gt, = Variable(img.cuda()), \
                                Variable(error.cuda()), \
                                Variable(score_gt.cuda())

        score_pred, senMap = model(img, error)
        # print(score_pred.data.cpu().numpy(), score_gt.data.cpu().numpy())
        # print(senMap.shape)

        score_pred_np = score_pred.data.cpu().numpy()
        score_gt_np = score_gt.data.cpu().numpy()

        print(batch_id, score_pred_np, score_gt_np)
        # img_ori, ref_ori = __find_test_img(score_gt_np)


        img_np = img.data.cpu().numpy()
        error_np = error.data.cpu().numpy()
        senMap_np = senMap.data.cpu().numpy()

        error_np = np.squeeze(error_np)
        img_np = np.squeeze(img_np)

        error_np_resize = skimage.measure.block_reduce(error_np, (4, 4), np.mean)

        perceptual = error_np_resize*senMap_np

        plt.figure(figsize=(12, 8))
        plt.suptitle('GT:%.4f, Predit:%.4f' % (score_gt_np, score_pred_np),
                     fontsize=16)

        plt.subplot(231)
        plt.imshow(img_ori)
        plt.xlabel('Distorted Img', fontsize=14)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()

        plt.subplot(234)
        plt.imshow(ref_ori)
        plt.xlabel('Reference Img', fontsize=14)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(232)
        plt.imshow(img_np)
        plt.xlabel('Per-processed Img', fontsize=14)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(233)
        plt.imshow(error_np_resize)
        plt.xlabel('Error Map', fontsize=14)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(235)
        plt.imshow(senMap_np)
        plt.xlabel('Sensitivity Map', fontsize=14)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(236)
        plt.imshow(perceptual)
        plt.xlabel('Perceptual Map', fontsize=14)
        plt.xticks([])
        plt.yticks([])

        plt.tight_layout()

        if batch_id == 4:
            plt.savefig('TID13_exp2.png', dpi=500)

        plt.show()

        if batch_id > 25:
            break


def showfigures_LIVE():
    testset = LIVEDatasetLoader('../data/LIVE_dataset/',
                                train_phase=False,
                                is_sample=False,
                                seed=12,
                                global_permute=False)
    testloader = DataLoader(testset,
                            shuffle=False,
                            batch_size=1,
                            num_workers=1,
                            pin_memory=True)

    # load the trained model
    model = predictNet()
    model_dict = torch.load('snapshots/deepQA_LIVE_seed12_0.9708_0.9665_epoch2430.pth')['model']
    model.load_state_dict(model_dict)  # copyWeights(model, model_dict, freeze=False)
    model.eval()
    model.to('cuda')

    for batch_id, (img, error, score_gt) in enumerate(testloader):
        score_gt = score_gt.type('torch.FloatTensor')
        img, error, score_gt, = Variable(img.cuda()), \
                                Variable(error.cuda()), \
                                Variable(score_gt.cuda())

        ### ml add
        

        score_pred, senMap = model(img, error)
        # print(score_pred.data.cpu().numpy(), score_gt.data.cpu().numpy())
        # print(senMap.shape)

        score_pred_np = score_pred.data.cpu().numpy()
        score_gt_np = score_gt.data.cpu().numpy()

        print(batch_id, score_pred_np, score_gt_np)

        img_np = img.data.cpu().numpy()
        error_np = error.data.cpu().numpy()
        senMap_np = senMap.data.cpu().numpy()

        error_np = np.squeeze(error_np)
        img_np = np.squeeze(img_np)

        error_np_resize = skimage.measure.block_reduce(error_np, (4, 4), np.mean)

        perceptual = error_np_resize*senMap_np

        plt.figure(figsize=(12, 8))
        plt.suptitle('GT:%.4f, Predit:%.4f' % (score_gt_np*10, score_pred_np*10),
                     fontsize=16)

        plt.subplot(231)
        plt.imshow(m.imread('../data/LIVE_dataset/studentsculpture/3img108-77.1967-GB.jpg'))
        plt.xlabel('Distorted Img', fontsize=14)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(234)
        plt.imshow(m.imread('../data/LIVE_dataset/studentsculpture/1studentsculptureOriginal.jpg'))
        plt.xlabel('Reference Img', fontsize=14)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(232)
        plt.imshow(img_np)
        plt.xlabel('Per-processed Img', fontsize=14)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(233)
        plt.imshow(error_np_resize)
        plt.xlabel('Error Map', fontsize=14)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(235)
        plt.imshow(senMap_np)
        plt.xlabel('Sensitivity Map', fontsize=14)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(236)
        plt.imshow(perceptual)
        plt.xlabel('Perceptual Map', fontsize=14)
        plt.xticks([])
        plt.yticks([])

        plt.tight_layout()

        if batch_id == 12:
            plt.savefig('LIVE_exp2.png', dpi=500)

        plt.show()

        if batch_id > 15:
            break



    



if __name__=='__main__':

    # showfigures_LIVE()

    showfigures_TID13()

    # test_momo = test_momo()
    # test_momo.preprocessing_vvzm()