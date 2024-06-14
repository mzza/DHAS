from functools import partial

from keras import backend as K
from keras.losses import binary_crossentropy
import math
from ..config_train import config


def dice_coefficient(y_true, y_pred,mask,smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    mask_f=K.flatten(mask)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection*mask_f + smooth) / ((K.sum(y_true_f) + K.sum(y_pred_f))*mask_f + smooth)

def binary_cross_entropy(y_true, y_pred,mask):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    mask_f=K.flatten(mask)
    epsilon = K.epsilon()  # 得到一个小的常数，用于避免log函数中的除零错误
    y_pred_f = K.clip(y_pred_f, epsilon, 1.0 - epsilon)  # 将预测值限制在一个小的范围内，避免log函数中的除零错误
    return -K.mean(y_true_f * K.log(y_pred_f)*mask_f + (1.0 - y_true_f) * K.log(1.0 - y_pred_f))

def dice_coefficient_loss(y_true, y_pred):
    return 1-dice_coefficient(y_true, y_pred)

def conditional_loss(y_true,y_pred):
    ## the whole region
   loss_whole=1-dice_coefficient(y_true[:,0],y_pred[:,0],mask=K.ones(shape=y_true[:,0].shape))\
              +binary_cross_entropy(y_true[:,0],y_pred[:,0],mask=K.ones(shape=y_true[:,0].shape))
   loss_core=1-dice_coefficient(y_true[:,1],y_pred[:,1],mask=y_true[:,0])\
             +binary_cross_entropy(y_true[:,1],y_pred[:,1],mask=y_true[:,0])
   loss_enhance=1-dice_coefficient(y_true[:,2],y_pred[:,2],mask=y_true[:,1])+\
                binary_cross_entropy(y_true[:,2],y_pred[:,2],mask=y_true[:,1])

   return loss_whole+loss_enhance+loss_core




def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(K.sum(y_true,
                                                            axis=axis) + K.sum(y_pred,
                                                                               axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    # if label_index==0:
    #     shape=y_true[:, label_index].shape
    #     current_mask=1;
    # else:
    #     current_mask=y_true[:, label_index-1]
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f

#  numerically stable unconditional probability (NSUP) loss
def NSUP_loss(y_true,y_pred,label_index):

    if label_index==0:
        y_true_f = K.flatten(y_true[:, label_index])
        y_pred_f = K.flatten(y_pred[:, label_index])
        last_term=-y_pred_f-K.logsumexp(-y_pred_f)
        loss=K.sum(last_term-last_term*y_true_f)
    if label_index==1:
        y_true_f0 = K.flatten(y_true[:, label_index-1])
        y_pred_f0 = K.flatten(y_pred[:, label_index-1])

        y_true_f1 = K.flatten(y_true[:, label_index])
        y_pred_f1 = K.flatten(y_pred[:, label_index])

        last_term=(-y_pred_f0-y_pred_f1)-(K.logsumexp(-y_pred_f0)+K.logsumexp(-y_pred_f1)+K.logsumexp(-y_pred_f0-y_pred_f1))
        loss=K.sum(last_term-last_term*y_true_f1)

    if label_index == 2:
        y_true_f0 = K.flatten(y_true[:, label_index - 2])
        y_pred_f0 = K.flatten(y_pred[:, label_index - 2])

        y_true_f1 = K.flatten(y_true[:, label_index-1])
        y_pred_f1 = K.flatten(y_pred[:, label_index-1])

        y_true_f2 = K.flatten(y_true[:, label_index])
        y_pred_f2 = K.flatten(y_pred[:, label_index])

        last_term = (-y_pred_f0 - y_pred_f1-y_pred_f2) - (
                    K.logsumexp(-y_pred_f0) + K.logsumexp(-y_pred_f1) +K.logsumexp(-y_pred_f2)+
                    K.logsumexp(-y_pred_f0 - y_pred_f1)+K.logsumexp(-y_pred_f0 - y_pred_f2)+K.logsumexp(-y_pred_f1 - y_pred_f2)+
                    K.logsumexp(-y_pred_f0 - y_pred_f1- y_pred_f2)
        )
        loss = K.sum(last_term - last_term * y_true_f2)


    return loss

def tree_triplit_loss_function(y_true,y_pred):
    sample_rate=20

    ## first select the grid pixels.
    grid_position=[math.floor(n*127/(sample_rate-1)) for n in range(sample_rate-1)]
    grid_truelabel=[]
    for b in range(config["batch_size"]):
      for i in grid_position:
          for j in grid_position:
              for k in grid_position:
                  grid_truelabel.append([b,x,y,z])


    coordinates = []
    # 遍历所有可能的点组合
    for i in range(len(grid_truelabel)):
        anchor=grid_truelabel[i]
        for j in range(len(grid_truelabel)):
            postive= grid_truelabel[j]
            for k in range(len(grid_truelabel)):
                neg=grid_truelabel[k]
                # 检查是否满足条件 f(x,y)<f(x,z)
                if tree_constrain(anchor,postive,neg):
                    # 将满足条件的点组合添加到列表中
                    coordinates.append(grid_truelabel[i]+grid_truelabel[j]+grid_truelabel[k])

    ## select hardest negtive
    all_samples=[]
    distance_list=[]
    #curr_coor=curr_coor=coordinates[0]
    pre_len=0
    for i in range(len(coordinates)):
        curr_coor=coordinates[i]
        an_value=y_pred[curr_coor[0],curr_coor[1],curr_coor[2],curr_coor[3],:]
        po_value = y_pred[curr_coor[4], curr_coor[5], curr_coor[6], curr_coor[7],:]
        ng_value = y_pred[curr_coor[8], curr_coor[9], curr_coor[10], curr_coor[11],:]
        d_an_po=get_tree_distance(an_value,po_value)
        d_an_ng=get_tree_distance(an_value,ng_value)
        distance_list.append([d_an_po,d_an_ng])

        if i<len(coordinates)-1:
            if (curr_coor[0],curr_coor[1],curr_coor[2],curr_coor[3])!=(coordinates[i+1][0],
                                                                           coordinates[i+1][1],
                                                                           coordinates[i+1][2],
                                                                           coordinates[i+1][3]):
                distance_list = np.array(distance_list)
                max_index = np.argmax(distance_list[:, 0])+pre_len
                min_index = np.argmin(distance_list[:, 1])+pre_len

                all_samples.append([curr_coor[0],curr_coor[1],curr_coor[2],curr_coor[3],
                                    coordinates[max_index][0],coordinates[max_index][1],coordinates[max_index][2],coordinates[max_index][3],
                                    coordinates[min_index][0],coordinates[min_index][1],coordinates[min_index][2],coordinates[min_index][3]])
                distance_list=[]
                pre_len = pre_len + distance_list.shape[0]


    loss=0
    for i in range(len(all_samples)):
        curr_coor = all_samples[i]
        an_value=y_pred[curr_coor[0],curr_coor[1],curr_coor[2],curr_coor[3],:]
        po_value = y_pred[curr_coor[4], curr_coor[5], curr_coor[6], curr_coor[7],:]
        ng_value = y_pred[curr_coor[8], curr_coor[9], curr_coor[10], curr_coor[11],:]
        d_an_po = get_tree_distance(an_value, po_value)
        d_an_ng = get_tree_distance(an_value, ng_value)

        m_value=(get_tree_label_distance(y_true[curr_coor[0],curr_coor[1],curr_coor[2],curr_coor[3],:],
                                        y_true[curr_coor[8], curr_coor[9], curr_coor[10], curr_coor[11],:])
                -get_tree_label_distance(y_true[curr_coor[0],curr_coor[1],curr_coor[2],curr_coor[3],:],
                                        y_true[curr_coor[4], curr_coor[5], curr_coor[6], curr_coor[7],:]))/3
        loss=loss+ max(d_an_po - d_an_ng + m_value, 0)


        return  loss




def get_tree_distance(A,B):
    dot_product = K.sum(A * B, axis=-1)

    # 计算 A 和 B 的模
    norm_A = K.sqrt(K.sum(K.square(A), axis=-1))
    norm_B = K.sqrt(K.sum(K.square(B), axis=-1))

    # 计算余弦相似度
    cosine_similarity = dot_product / (norm_A * norm_B + K.epsilon())

    return 0.5*(1-cosine_similarity)

def get_tree_label_distance(x,y):
    if x==0&y==1:
        return 4
    if x==0&y==2:
        return 3
    if x==0&y==3:
        return 4
    if x==1&y==0:
        return 4
    if x==1&y==2:
        return 3
    if x==1&y==3:
        return 2
    if x==2&y==0:
        return 3
    if x==2&y==1:
        return 3
    if x==2&y==3:
        return 3
    if x==3&y==0:
        return 4
    if x==3&y==1:
        return 2
    if x==3&y==2:
        return 3

# NET=1,Edema=2. ET=3
def tree_constrain(b,a,p,n):
    ## the same class
    if a==3&p==3&n==1:
        return 1
    elif a==3&p==3&n==2:
        return 1
    elif a==3&p==3&n==0:
        return 1
    elif a==1&p==1&n==3:
        return 1
    elif a==1&p==1&n==2:
        return 1
    elif a==1&p==1&n==0:
        return 1
    elif a==2&p==2&n==3:
        return 1
    elif a==2&p==2&n==1:
        return 1
    elif a==2&p==2&n==0:
        return 1
    elif a==0&p==0&n==3:
        return 1
    elif a==0&p==0&n==2:
        return 1
    elif a==0&p==0&n==1:
        return 1
 ## fifferent class
    elif a==3&p==1&n==2:
        return 1
    elif a==3&p==1&n==0:
        return 1
    elif a==3&p==2&n==0:
        return 1
    elif a==1&p==3&n==2:
        return 1
    elif a==1&p==3&n==0:
        return 1
    elif a==3&p==1&n==0:
        return 1
    else:
        return 0




def cosine_annealing(epoch, initial_x, T):
    return 0.5 * (1 + math.cos(math.pi * (epoch % T) / T)) * initial_x

dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss
