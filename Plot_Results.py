import numpy as np
import warnings
from matplotlib import pylab
from matplotlib.patches import Patch
from prettytable import PrettyTable
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'AOA-RA-ACCGAN', 'AZO-RA-ACCGAN', 'CWO-RA-ACCGAN', 'OPA-RA-ACCGAN', 'ASOPA-RA-ACCGAN']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for i in range(Fitness.shape[0]):
        Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
        for j in range(len(Algorithm) - 1):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Analysis  ',
              '--------------------------------------------------')
        print(Table)

        length = np.arange(Fitness.shape[2])
        fig = plt.figure()
        fig.canvas.manager.set_window_title(' Convergence Curve')
        Conv_Graph = Fitness[0]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label=Algorithm[1])
        plt.plot(length, Conv_Graph[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label=Algorithm[2])
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label=Algorithm[3])
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label=Algorithm[4])
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label=Algorithm[5])
        plt.xlabel('No. of Iteration', fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.ylabel('Cost Function', fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.legend(loc=1, prop={'weight': 'bold', 'size': 12})
        plt.savefig("./Results/Conv_%s.png" % (i + 1))
        plt.show()


def Table():
    eval = np.load('Evaluate.npy', allow_pickle=True)
    Terms = ['MEP', 'SMAPE', 'MASE', 'MAE', 'RMSE', 'MSE', 'Accuracy']
    Algorithm = ['Optimizer', 'AOA-ST-ALSTM', 'AZO-ST-ALSTM', 'CWO-ST-ALSTM', 'OPA-ST-ALSTM', 'ASOPA-ST-ALSTM']
    Classifier = ['Optimizer', 'CNN', 'InceptionV3', 'LSTM', 'ST-LSTM ', 'ASOPA-ST-ALSTM']
    Graph_Term = np.array([2, 3, 4, 5, 6]).astype(int)
    Table_Terms = [2, 3, 4, 5, 6]
    table_terms = [Terms[i] for i in Table_Terms]
    Optimizer = ['Adam', 'SGD', 'RMSprop', 'Adamax', 'AdaGrad']
    for i in range(eval.shape[0]):
        for k in range(len(Table_Terms)):
            value = eval[i, :, :, :]
            Table = PrettyTable()
            Table.add_column(Classifier[0], Optimizer)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, j, Graph_Term[k]])
            print('--------------------------------Dataset - ', i + 1, table_terms[k], '  Algorithm Comparison',
                  '---------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], Optimizer)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Term[k]])
            print('-------------------------Dataset - ', i + 1, table_terms[k], '  Classifier Comparison',
                  '---------------------------------------')
            print(Table)


def Plot_Results():
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['MEP', 'SMAPE', 'MASE', 'MAE', 'RMSE', 'MSE', 'Accuracy']
    Algorithm = ['AOA-ST-ALSTM', 'AZO-ST-ALSTM', 'CWO-ST-ALSTM', 'OPA-ST-ALSTM', 'ASOPA-ST-ALSTM']
    Classifier = ['CNN', 'InceptionV3', 'LSTM', 'ST-LSTM ', 'ASOPA-ST-ALSTM']
    Graph_Terms = [2, 3, 4, 5, 6]
    Epoch = [20, 40, 60, 80]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j]]

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
            fig.canvas.manager.set_window_title('Dataset-' + str(i + 1) + ' Algorithm  Comparison of Epochs')
            X = np.arange(len(Epoch))
            width = 0.08
            ax.bar(X + 0.00 + width / 4, Graph[:, 0], color='darkgreen', edgecolor='w', width=0.15,
                   label=Algorithm[0])
            ax.bar(X + 0.15 + width / 4, Graph[:, 1], color='darkorange', edgecolor='w', width=0.15, label=Algorithm[1])
            ax.bar(X + 0.30 + width / 4, Graph[:, 2], color='#9A32CD', edgecolor='w', width=0.15,
                   label=Algorithm[2])
            ax.bar(X + 0.45 + width / 4, Graph[:, 3], color='#A52A2A', edgecolor='w', width=0.15, label=Algorithm[3])
            ax.bar(X + 0.60 + width / 4, Graph[:, 4], color='k', edgecolor='w', width=0.15, label=Algorithm[4])
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for color
                           in ['darkgreen', 'darkorange', '#9A32CD', '#A52A2A', 'k']]
            plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.18), fontsize=10,
                       frameon=False, ncol=3, prop={'weight': 'bold', 'size': 12})
            plt.xticks(X + 0.32, ('20', '40', '60', '80'), fontname="Arial", fontsize=14,
                       fontweight='bold')
            plt.xlabel('No. of Epochs', fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=14, fontweight='bold',
                       color='#35530a')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            path = "./Results/Dataset_%s_%s_Alg_bar.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
            fig.canvas.manager.set_window_title('Dataset-' + str(i + 1) + ' Method  Comparison of Epochs')
            X = np.arange(len(Epoch))
            width = 0.08
            ax.bar(X + 0.00 + width / 4, Graph[:, 5], color='m', edgecolor='w', width=0.15,
                   label=Classifier[0])
            ax.bar(X + 0.15 + width / 4, Graph[:, 6], color='y', edgecolor='w', width=0.15, label=Classifier[1])
            ax.bar(X + 0.30 + width / 4, Graph[:, 7], color='#9b5de5', edgecolor='w', width=0.15,
                   label=Classifier[2])
            ax.bar(X + 0.45 + width / 4, Graph[:, 8], color='#218380', edgecolor='w', width=0.15, label=Classifier[3])
            ax.bar(X + 0.60 + width / 4, Graph[:, 4], color='k', edgecolor='w', width=0.15, label=Classifier[4])
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for color
                           in ['m', 'y', '#9b5de5', '#218380', 'k']]
            plt.legend(dot_markers, Classifier, loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=10,
                       frameon=False, ncol=3, prop={'weight': 'bold', 'size': 12})
            plt.xticks(X + 0.27, ('20', '40', '60', '80'), fontname="Arial", fontsize=14,
                       fontweight='bold')
            plt.xlabel('No. of Epochs', fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=14, fontweight='bold',
                       color='#35530a')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            path = "./Results/Datset_%s_%s_Mod_bar.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def Proposed_Mod_Results():
    eval = np.load('Evaluate_Kfold.npy', allow_pickle=True)
    Terms = ['MEP', 'SMAPE', 'MASE', 'MAE', 'RMSE', 'MSE', 'Accuracy']
    Graph_Terms = [2, 3, 4, 5, 6]
    Classifier = np.asarray(['ST-LSTM ', 'ASOPA-ST-ALSTM'])
    for n in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graphs = eval[n, :, :, Graph_Terms[j]]
            with_optimiztion = Graphs[:, 4]
            without_optimization = Graphs[:, 8]
            dark_colors = ['#254C78']
            light_colors = ['#AEC6CF']
            kfold = [1, 2, 3, 4, 5]
            X = np.arange(len(kfold))
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('white')  # White figure background
            ax.set_facecolor('white')  # White plot background
            ax.xaxis.grid(True, color='#e0e0e0', linewidth=1)
            ax.set_axisbelow(True)
            bar1 = ax.barh(X, with_optimiztion, color=dark_colors, height=0.6)
            ax.bar_label(container=bar1, size=9, label_type='edge', labels=[f'{x:.2f}' for x in Graphs[:, 4]],
                         fontweight='bold', color='w', padding=-35)
            bar2 = ax.barh(X, without_optimization, color=light_colors, height=0.4)
            ax.bar_label(container=bar2, size=9, label_type='edge', labels=[f'{x:.2f}' for x in Graphs[:, 8]],
                         fontweight='bold', color='k', padding=6)
            plt.yticks(X, ['1', '2', '3', '4', '5'],
                       fontname="Arial", fontsize=14,
                       fontweight='bold', color='k')
            plt.xlabel(Terms[Graph_Terms[j]], fontweight='bold', fontsize=14)
            plt.ylabel('KFold', fontweight='bold', fontsize=14)
            plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
            # ax.set_xlim(0, np.max(Graphs) * 1.1)
            ax.invert_yaxis()

            legend_elements = [
                Patch(facecolor='#AEC6CF', label=Classifier[0]),
                Patch(facecolor='#254C78', label=Classifier[1])
            ]
            ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.19), ncol=2, fontsize=12,
                      frameon=False, prop={'weight': 'bold', 'size': 12})
            plt.tight_layout(rect=[0, 0.05, 1, 1])

            path = f"./Results/Dataset_Kfold {n + 1, Terms[Graph_Terms[j]]} Method.png"
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title(f" K fold {Terms[Graph_Terms[j]]}")
            plt.tight_layout()
            plt.savefig(path)
            plt.show()


def Proposed_Alg_Results():
    eval = np.load('Evaluate_Kfold.npy', allow_pickle=True)
    Terms = ['MEP', 'SMAPE', 'MASE', 'MAE', 'RMSE', 'MSE', 'Accuracy']
    Graph_Terms = [2, 3, 4, 5, 6]
    Classifier = np.asarray(['OPA-ST-ALSTM', 'ASOPA-ST-ALSTM'])
    for n in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graphs = eval[n, :, :, Graph_Terms[j]]
            with_optimiztion = Graphs[:, 4]
            without_optimization = Graphs[:, 3]
            dark_colors = ['#d62728']
            light_colors = ['#ff9896']
            kfold = [1, 2, 3, 4, 5]
            X = np.arange(len(kfold))
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('white')  # White figure background
            ax.set_facecolor('white')  # White plot background
            ax.xaxis.grid(True, color='#e0e0e0', linewidth=1)
            ax.set_axisbelow(True)
            bar1 = ax.barh(X, with_optimiztion, color=dark_colors, height=0.6)
            ax.bar_label(container=bar1, size=9, label_type='edge', labels=[f'{x:.2f}' for x in Graphs[:, 4]],
                         fontweight='bold', color='k', padding=6)
            bar2 = ax.barh(X, without_optimization, color=light_colors, height=0.4)
            ax.bar_label(container=bar2, size=9, label_type='edge', labels=[f'{x:.2f}' for x in Graphs[:, 3]],
                         fontweight='bold', color='w', padding=-35)

            plt.yticks(X, ['1', '2', '3', '4', '5'],
                       fontname="Arial", fontsize=12,
                       fontweight='bold', color='k')
            plt.xlabel(Terms[Graph_Terms[j]], fontweight='bold', fontsize=14)
            plt.ylabel('KFold', fontweight='bold', fontsize=14)
            plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
            ax.invert_yaxis()

            legend_elements = [
                Patch(facecolor='#ff9896', label=Classifier[0]),
                Patch(facecolor='#d62728', label=Classifier[1])
            ]
            ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.19), ncol=2, fontsize=12,
                      frameon=False, prop={'weight': 'bold', 'size': 12})
            plt.tight_layout(rect=[0, 0.05, 1, 1])

            path = f"./Results/Dataset_Kfold {n + 1, Terms[Graph_Terms[j]]} Alg.png"
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title(f"K fold {Terms[Graph_Terms[j]]}")
            plt.tight_layout()
            plt.savefig(path)
            plt.show()


def plot_seg_results():
    eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'IOU', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV',
             'FDR', 'F1-Score', 'MCC']
    Classifier = ['Unet', 'ResNet', 'DenseNet', 'PSPN', 'DPSPN-RDN']
    Labels = np.asarray(['Ref 6', 'Proposed'])
    Epoch = ['Linear', 'ReLu', 'Softmax', 'Sigmoid', 'Thnh']
    Epoch = np.asarray(Epoch)
    Graph_Terms = [0, 1, 2]
    colors = ['r', 'm', 'y', 'brown', 'k']
    for n in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = eval[n, :, :, 0, Graph_Terms[j] + 4]

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            fig.canvas.manager.set_window_title('Method Comparison of Activation Function')
            X = np.arange(len(Epoch) - 2)
            plt.barh(X + 0.00, Graph[:3, 0], color='r', height=0.15, label=Classifier[0])
            plt.barh(X + 0.15, Graph[:3, 1], color='m', height=0.15, label=Classifier[1])
            plt.barh(X + 0.30, Graph[:3, 2], color='y', height=0.15, label=Classifier[1])
            plt.barh(X + 0.45, Graph[:3, 3], color='brown', height=0.15, label=Classifier[1])
            plt.barh(X + 0.60, Graph[:3, 4], color='k', height=0.15, label=Classifier[1])
            plt.yticks(X + 0.15, ['Linear', 'ReLu', 'Softmax'], fontname="Arial", fontsize=14,
                       fontweight='bold', color='k')
            plt.ylabel('Activation Function', fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.xlabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(False)

            dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=12) for color
                           in colors]
            plt.legend(dot_markers, Classifier, loc='upper center', bbox_to_anchor=(0.5, 1.10), fontsize=12,
                       frameon=False, ncol=len(Classifier), prop={'weight': 'bold', 'size': 12})
            plt.tight_layout()
            plt.grid(axis='x')
            path = "./Results/Dataset_%s_%s_Seg_bar.png" % (n + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def Fusion_Plots_Results():
    eval = np.load('Evaluate_fusion.npy', allow_pickle=True)
    Terms = ['BRISQUE', 'NIQE', 'PIQE', 'PSNR', 'FID', 'ID']
    Graph_Terms = [0, 1, 2, 3, 4, 5]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            LR = ['Learning Rate - 0.55', 'Learning Rate - 0.66', 'Learning Rate - 0.77']
            Algorithm = ['AOA-RA-ACCGAN', 'AZO-RA-ACCGAN', 'CWO-RA-ACCGAN', 'OPA-RA-ACCGAN', 'ASOPA-RA-ACCGAN']
            Classifier = ['Cycle GAN', 'Conditional GAN', 'Convolution GAN', 'CCGAN ', 'RA-ACCGAN']
            Graph = eval[i, :, :5, Graph_Terms[j]]
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_axes([0.25, 0.12, 0.7, 0.7])
            fig.canvas.manager.set_window_title('Algorithm Comparison of Learning Rate')
            X = np.arange(len(Algorithm))
            bar3 = plt.barh(X + 0.00, Graph[0, :], color='#8ecae6', height=0.30,
                            label=Algorithm[0])
            ax.bar_label(container=bar3, size=9, label_type='edge', labels=[f'{x:.1f}' for x in Graph[0, :]],
                         fontweight='bold', color='w', padding=-35)
            bar4 = plt.barh(X + 0.30, Graph[1, :], color='#fcbf49', height=0.30,
                            label=Algorithm[1])
            ax.bar_label(container=bar4, size=9, label_type='edge', labels=[f'{x:.1f}' for x in Graph[1, :]],
                         fontweight='bold', color='w', padding=-35)
            bar5 = plt.barh(X + 0.60, Graph[2, :], color='#588157', height=0.30,
                            label=Algorithm[2])
            ax.bar_label(container=bar5, size=9, label_type='edge', labels=[f'{x:.1f}' for x in Graph[2, :]],
                         fontweight='bold', color='w', padding=-35)
            plt.yticks(X + 0.20, Algorithm, fontsize=14,
                       fontname="Arial",
                       fontweight='bold', color='#14213d')
            plt.xlabel(Terms[Graph_Terms[j]], fontsize=14, fontname="Arial", fontweight='bold', color='k')
            plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='#14213d')
            plt.gca().spines['top'].set_visible(True)
            plt.gca().spines['right'].set_visible(True)
            plt.gca().spines['left'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(True)
            dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=12) for color
                           in ['#8ecae6', '#fcbf49', '#588157']]
            plt.legend(dot_markers, LR, loc='upper center', bbox_to_anchor=(0.48, 1.25), fontsize=10,
                       frameon=False, ncol=2, prop={'weight': 'bold', 'size': 14})
            plt.tight_layout()
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            path = "./Results/Dataset_%s_%s_Alg_ImageGeneration_bar.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()

            Graph = eval[i, 2:, 5:, Graph_Terms[j]]
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_axes([0.25, 0.12, 0.7, 0.7])
            fig.canvas.manager.set_window_title('Algorithm Comparison of Learning Rate')
            X = np.arange(len(Classifier))
            bar7 = plt.barh(X + 0.00, Graph[0, :], color='#ffb5a7', height=0.30, label=Classifier[0])
            ax.bar_label(container=bar7, size=9, label_type='edge', labels=[f'{x:.1f}' for x in Graph[0, :]],
                         fontweight='bold', color='w', padding=-35)
            bar8 = plt.barh(X + 0.30, Graph[1, :], color='#00afb9', height=0.30, label=Classifier[1])
            ax.bar_label(container=bar8, size=9, label_type='edge', labels=[f'{x:.1f}' for x in Graph[1, :]],
                         fontweight='bold', color='w', padding=-35)
            bar9 = plt.barh(X + 0.60, Graph[2, :], color='#400040', height=0.30, label=Classifier[2])
            ax.bar_label(container=bar9, size=9, label_type='edge', labels=[f'{x:.1f}' for x in Graph[2, :]],
                         fontweight='bold', color='w', padding=-35)
            plt.yticks(X + 0.20, Classifier, fontname="Arial",
                       fontsize=14,
                       fontweight='bold', color='#14213d')
            plt.xlabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='#14213d')
            plt.gca().spines['top'].set_visible(True)
            plt.gca().spines['right'].set_visible(True)
            plt.gca().spines['left'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(True)
            dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=12) for color
                           in ['#ffb5a7', '#00afb9', '#400040']]
            plt.legend(dot_markers, LR, loc='upper center', bbox_to_anchor=(0.48, 1.25), fontsize=10,
                       frameon=False, ncol=2, prop={'weight': 'bold', 'size': 14})
            plt.tight_layout()
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            path = "./Results/Dataset_%s_%s_mod_ImageGeneration_bar.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def Seg_Table():
    eval = np.load('Eval_Opti_all.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'IOU', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV',
             'FDR', 'F1-Score', 'MCC']
    Classifier = ['Optimizer', 'Unet', 'ResNet', 'DenseNet', 'PSPN', 'DPSPN-RDN']
    Graph_Term = np.array([0, 1, 2]).astype(int)
    Table_Terms = [0, 1, 2]
    table_terms = [Terms[i] for i in Table_Terms]
    Optimizer = ['Adam', 'SGD', 'RMSprop', 'Adamax', 'AdaGrad']
    for i in range(eval.shape[0]):
        for k in range(len(Table_Terms)):
            value = eval[i, :, :, 4:]
            Table = PrettyTable()
            Table.add_column(Classifier[0], Optimizer)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, j, Graph_Term[k]])
            print('-------------------------Dataset - ', i + 1, table_terms[k], '  Segmentation Comparison',
                  '---------------------------------------')
            print(Table)


if __name__ == '__main__':
    plotConvResults()
    Plot_Results()
    Table()
    Proposed_Alg_Results()
    Proposed_Mod_Results()
    plot_seg_results()
    Fusion_Plots_Results()
    Seg_Table()
