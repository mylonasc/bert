
import itertools
import numpy as np
import os
import pdb


import matplotlib as mpl
#mpl.use('Agg')

import matplotlib.pyplot as plt


#from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, saveat = None,suff = None, figsize = None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    if figsize is not None:
        plt.figure(figsize = figsize)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize = 7)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if saveat is not None:
        plt.savefig(os.path.join(saveat  , 'confussion_matrix' + suff + '.png'))

        plt.cla()
        plt.close()

    

def plot_attention_matrix(cm, classes,
                          title='Attention matrix',
                          cmap=plt.cm.Greens, saveat = None,suff = None, figsize = None, hide_special = False):
    """
    This function prints and plots the attention matrix.
    """

    if hide_special:
        dd = [t for t in zip(classes , range(0, len(classes))) if t[0] != '[SEP]' and t[0] !='[CLS]'] # get the indices of the tokens and the stripped tokens.
        classes = [d[0] for d in dd]
        ids = [d[1] for d in dd];
        cm = cm[ids].T[ids].T # taking out the columns and rows that correspond to the indices with special tokens.
    else:
        # We hide only the attention to outputs that correspond to empty tokens.
        r = range(0,len(classes));
        cm = cm[r].T[r].T
        #pdb.set_trace()



    if figsize is not None:
        plt.figure(figsize = figsize)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.ylabel('tokens')
    plt.xlabel('tokens')
    plt.tight_layout()
    if saveat is not None:
        plt.savefig(os.path.join(saveat  , 'attention_matrix' + suff + '.png'))

        plt.cla()
        plt.close()

def plot_attention_words(cm, text_1 ,text_2= None, color = None , figsize = None):
    """
    Plot attention
    """
    if text_2 == None: 
        text_2 = text_1

    fig = plt.figure(figsize = figsize)

    ax = fig.add_subplot(111)

    offs = 0;
    offs_step = 0.1
    for k in text_1:
        ax.text(0+offs,0.3, k, rotation = 90, fontsize = 24)
        offs = offs + offs_step

    offs = 0;
    for m in text_2:
        ax.text(offs ,0.65, m, rotation = 90, fontsize = 24, horizontalalignment = 'left', verticalalignment = 'bottom')
        offs = offs + offs_step


    ax.axis('off')

    x_vals= np.linspace(0,offs-offs_step,len(text_1))+0.05
    y_start = 0.4
    y_end  = 0.6
    for k in range(0,cm.shape[0]):
        for m in range(0,cm.shape[1]):
            line = mpl.lines.Line2D([x_vals[k], x_vals[m]], [y_start, y_end], lw = 5,alpha = 1 * cm[k,m], color = None)
            ax.add_line(line)

