import os
import cv2
from numpy import matlib
from AOA import AOA
from AZO import AZO
from CWO import CWO
from Gen_Mask import Gen_Mask
from Global_Vars import Global_Vars
from Image_Results import *
from Model_CGAN import Model_CGAN
from Model_CNN import Model_CNN
from Model_Con_GAN import Model_Con_GAN
from Model_Conv_GAN import Model_Conv_GAN
from Model_DenseNet import Model_DenseNet
from Model_FCN import Model_FCN
from Model_GAN import Model_GAN
from Model_InceptionV3 import Model_InceptionV3
from Model_RAN import Model_RAN
from Model_RDNet import Model_RDNet
from Model_ST_LSTM import Model_ST_LSTM
from Model_Unet import Model_Unet
from Model_Unetplusplus import Model_Unetplusplus
from OPA import OPA
from Objfun import objfun_1, objfun_2
from Plot_Results import *
from Proposed import Proposed
import warnings

warnings.filterwarnings('ignore')

No_of_Dataset = 2

# Read Dataset 1
an = 0
if an == 1:
    Image = []
    GroundTruth = []
    Path = './Dataset/Dataset1/train/img'
    Label_Path = './Dataset1/train/label'
    In_dir = os.listdir(Path)
    label_dir = os.listdir(Label_Path)
    for i in range(len(In_dir)):
        FileName = Path + '/' + In_dir[i]
        LabelName = Label_Path + '/' + In_dir[i]
        Img = cv.imread(FileName)
        label = cv.imread(LabelName)
        Re_Img = cv.resize(Img, (512, 512))
        Re_label = cv.resize(label, (512, 512))
        Image.append(Re_Img)
        GroundTruth.append(Re_label)
    np.save('Image_1.npy', np.asarray(Image))
    np.save('GroundTruth_1.npy', np.asarray(GroundTruth))

# Read Dataset 2
an = 0
if an == 1:
    Image = []
    Path = './Dataset/Dataset2/train/images'
    In_dir = os.listdir(Path)
    for i in range(len(In_dir)):
        FileName = Path + '/' + In_dir[i]
        Img = cv.imread(FileName)
        Re_Img = cv.resize(Img, (512, 512))
        Image.append(Re_Img)
    np.save('Image_2.npy', np.asarray(Image))

# Generate GroundTruth for Dataset 2
an = 0
if an == 1:
    Image = np.load('Image_2.npy', allow_pickle=True)
    Mask = Gen_Mask(Image)
    np.save('GroundTruth_2.npy', Mask)

# Optimization for Synthetic Image Generation
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Image = np.load('Image_' + str(n+1) + '.npy', allow_pickle=True)
        Best_sol = []
        Global_Vars.Data = Image
        Npop = 10
        Chlen = 3  # Hidden Neuron Count, Learning Rate and Activation Function
        xmin = matlib.repmat([5, 0.01, 1], Npop, 1)
        xmax = matlib.repmat([255, 0.99, 5], Npop, 1)
        initsol = np.zeros(xmin.shape)
        for i in range(xmin.shape[0]):
            for j in range(xmin.shape[1]):
                initsol[i, j] = np.random.uniform(xmin[i, j], xmax[i, j])
        fname = objfun_1
        max_iter = 50

        print('AOA....')
        [bestfit1, fitness1, bestsol1, Time1] = AOA(initsol, fname, xmin, xmax, max_iter)

        print('AZO....')
        [bestfit2, fitness2, bestsol2, Time2] = AZO(initsol, fname, xmin, xmax, max_iter)

        print('CWO....')
        [bestfit3, fitness3, bestsol3, Time3] = CWO(initsol, fname, xmin, xmax, max_iter)

        print('OPA....')
        [bestfit4, fitness4, bestsol4, Time4] = OPA(initsol, fname, xmin, xmax, max_iter)

        print('PROPOSED....')
        [bestfit5, fitness5, bestsol5, Time5] = Proposed(initsol, fname, xmin, xmax, max_iter)

        BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        np.save('BestSol_' + str(n + 1) + '.npy', BestSol)

# Synthetic Image Generation
an = 0
if an == 1:
    Eval_all = []
    for n in range(No_of_Dataset):
        Image = np.load('Image_' + str(n + 1) + '.npy', allow_pickle=True)
        BestSol = np.load('BestSol_' + str(n + 1) + '.npy', allow_pickle=True)
        LearningRate = [0.55, 0.66, 0.77]
        EVAL = []
        Eff_Image = []
        for act in range(len(LearningRate)):
            Eval = np.zeros((10, 6))
            for j in range(BestSol.shape[0]):
                sol = np.round(BestSol[j, :]).astype(np.int16)
                Eval[j, :], pred = Model_GAN(Image, LearningRate[act], sol)  # With optimization
            Eval[5, :], pred1 = Model_CGAN(Image, LearningRate[act])
            Eval[6, :], pred2 = Model_Con_GAN(Image, LearningRate[act])
            Eval[7, :], pred3 = Model_Conv_GAN(Image, LearningRate[act])
            Eval[8, :], pred4 = Model_GAN(Image, LearningRate[act])
            Eval[9, :] = Eval[4, :]
            EVAL.append(Eval)
            Eff_Image.append(pred4)
        Eval_all.append(EVAL)
        np.save('Synthetic_Image_' + str(n + 1) + '.npy', Eff_Image)
    np.save('Evaluate_fusion.npy', Eval_all)

# Segmentation
an = 0
if an == 1:
    Eval_all = []
    for n in range(No_of_Dataset):
        Image = np.load('Synthetic_Image_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('GroundTruth_' + str(n + 1) + '.npy', allow_pickle=True)
        EVAL = []
        Act_fun = ['Linear', 'ReLu', 'Softmax', 'Sigmoid', 'TanH']
        Seg = []
        for act in range(len(Act_fun)):
            Eval = np.zeros((5, 11))
            Eval[0, :], pred1 = Model_Unet(Image, Target, Act_fun[act])
            Eval[1, :], pred2 = Model_Unetplusplus(Image, Target, Act_fun[act])
            Eval[2, :], pred3 = Model_FCN(Image, Target, Act_fun[act])
            Eval[3, :], pred4 = Model_DenseNet(Image, Target, Act_fun[act])
            Eval[4, :], SegImage = Model_RDNet(Image, Target, Act_fun[act])
            EVAL.append(Eval)
            Seg.append(SegImage)
        Eval_all.append(EVAL)
        np.save('Seg_RDNet_' + str(n + 1) + '.npy', Seg)
    np.save('Eval_all.npy', Eval_all)

# Generate Target
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Image = np.load('Seg_RDNet_' + str(n + 1) + '.npy', allow_pickle=True)
        Tar = []
        for s in range(len(Image)):
            img = Image[s]
            orig = img.copy()
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower = np.array([0, 0, 0])
            upper = np.array([180, 255, 80])
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.medianBlur(mask, 7)
            mask = cv2.GaussianBlur(mask, (9, 9), 0)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                print("Vehicle not found")
                exit()
            vehicle = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(vehicle)
            cx = x + w // 2
            cy = y + h // 2
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(orig, (cx, cy), 5, (0, 0, 255), -1)
            target_y = cy - 150
            if target_y < 0:
                target_y = 0
            target_point = (cx, target_y)
            Tar.append(target_point)
        np.save('Target_' + str(n + 1) + '.npy', np.asarray(Tar))

# Optimization for prediction
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Image = np.load('Seg_RDNet_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Best_sol = []
        Global_Vars.Data = Image
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 3  # Hidden Neuron Count, Learning Rate and Activation Function
        xmin = matlib.repmat([5, 0.01, 1], Npop, 1)
        xmax = matlib.repmat([255, 0.99, 5], Npop, 1)
        initsol = np.zeros(xmin.shape)
        for i in range(xmin.shape[0]):
            for j in range(xmin.shape[1]):
                initsol[i, j] = np.random.uniform(xmin[i, j], xmax[i, j])
        fname = objfun_2
        max_iter = 50

        print('AOA....')
        [bestfit1, fitness1, bestsol1, Time1] = AOA(initsol, fname, xmin, xmax, max_iter)

        print('AZO....')
        [bestfit2, fitness2, bestsol2, Time2] = AZO(initsol, fname, xmin, xmax, max_iter)

        print('CWO....')
        [bestfit3, fitness3, bestsol3, Time3] = CWO(initsol, fname, xmin, xmax, max_iter)

        print('OPA....')
        [bestfit4, fitness4, bestsol4, Time4] = OPA(initsol, fname, xmin, xmax, max_iter)

        print('PROPOSED....')
        [bestfit5, fitness5, bestsol5, Time5] = Proposed(initsol, fname, xmin, xmax, max_iter)

        BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        np.save('Sol_' + str(n + 1) + '.npy', BestSol)

# Prediction
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Image = np.load('Seg_RDNet_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        BestSol = np.load('Sol_' + str(n + 1) + '.npy', allow_pickle=True)
        EVAL = []
        Epoch = [20, 40, 60, 80]
        for act in range(len(Epoch)):
            learnperc = round(Image.shape[0] * 0.75)  # Split Training and Testing Datas
            Train_Data = Image[:learnperc, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Image[learnperc:, :]
            Test_Target = Target[learnperc:, :]
            Eval = np.zeros((10, 25))
            for j in range(BestSol.shape[0]):
                print(act, j)
                sol = np.round(BestSol[j, :]).astype(np.int16)
                Eval[j, :], pred = Model_ST_LSTM(Train_Data, Train_Target, Test_Data, Test_Target, Epoch[act],
                                                 sol)  # With Optimization
            Eval[5, :], pred1 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target,
                                          Epoch[act])
            Eval[6, :], pred2 = Model_RAN(Train_Data, Train_Target, Test_Data,
                                          Test_Target, Epoch[act])
            Eval[7, :], pred3 = Model_InceptionV3(Train_Data, Train_Target, Test_Data, Test_Target,
                                                  Epoch[act])
            Eval[8, :], pred4 = Model_ST_LSTM(Train_Data, Train_Target, Test_Data, Test_Target,
                                              Epoch[act])  # Without Optimization
            Eval[9, :] = Eval[4, :]
            EVAL.append(Eval)
        np.save('Evaluate_all.npy', EVAL)  # Save Eval

plotConvResults()
Plot_Results()
Table()
Proposed_Alg_Results()
Proposed_Mod_Results()
plot_seg_results()
Fusion_Plots_Results()
Image_Results()
Sample_Images()
