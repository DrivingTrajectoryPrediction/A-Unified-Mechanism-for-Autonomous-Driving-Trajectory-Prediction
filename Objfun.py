import numpy as np

from Evaluate_Error import Image_Evaluation, evaluat_error
from Global_Vars import Global_Vars
from Model_GAN import Model_GAN
from Model_ST_LSTM import Model_ST_LSTM


def objfun_1(Soln):
    data = Global_Vars.Data
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Eval, Image = Model_GAN(data, sol)
            Eval = Image_Evaluation(data[0], Image[0])
            Fitn[i] = Eval[1]
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        Eval, Image = Model_GAN(data, sol)
        Eval = Image_Evaluation(data[0], Image[0])
        Fitn = Eval[1]
        return Fitn


def objfun_2(Soln):
    data = Global_Vars.Data
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(data.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Train_Data = data[:learnper, :]
            Train_Target = Tar[:learnper, :]
            Test_Data = data[learnper:, :]
            Test_Target = Tar[learnper:, :]
            Eval, pred = Model_ST_LSTM(Train_Data, Train_Target, Test_Data, Test_Target, 20, sol)
            Eval = evaluat_error(pred, Test_Target)
            Fitn[i] = 1 / Eval[6]
        return Fitn
    else:
        learnper = round(data.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = data[:learnper, :]
        Train_Target = Tar[:learnper, :]
        Test_Data = data[learnper:, :]
        Test_Target = Tar[learnper:, :]
        Eval, pred = Model_ST_LSTM(Train_Data, Train_Target, Test_Data, Test_Target, 20, sol)
        Eval = evaluat_error(pred, Test_Target)
        Fitn = 1 / Eval[6]

    return Fitn
