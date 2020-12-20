#!/usr/bin/env python
# coding: utf-8

# ### 采用wide&deep 模型进行rank

# In[4]:


import tensorflow as tf

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss

from tensorflow.keras.layers import *
import tensorflow.keras.backend as K 
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import *
from tensorflow.keras.regularizers import l2

import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from collections import Counter, OrderedDict, namedtuple
import warnings
warnings.filterwarnings("ignore")


# In[5]:


print('tensorflow version: ', tf.__version__)


# In[4]:


if __name__ == '__main__':
    with open('./feature_all_train_test.pkl', 'rb') as f:
        feature_all_train_test = pickle.load(f)

    feature_all_train_test.head()


# In[4]:


if __name__ == "__main__":
    print('----------- feature加载完成 -----------')
    feature_all_train_test.shape


# ### 抽样构建多个数据集

# ### 将召回列表中真正发生点击的用户-商品对视为正样，按1:5的正负比例从召回列表中随机选取负样，生成6个数据集

# In[5]:


if __name__ == "__main__":
    all_test = feature_all_train_test[feature_all_train_test['train_flag'] == 'test'].drop('train_flag', axis=1)
    all_train = feature_all_train_test[feature_all_train_test['train_flag'] == 'train'].drop(['train_flag', 'item_similar', 'user_id'], axis=1)
    del feature_all_train_test


# In[ ]:


# In[7]:


if __name__ == "__main__":
    all_train_positive = all_train[all_train['label'] == 1].reset_index(drop=True)
    all_train_negative = all_train[all_train['label'] == 0].reset_index(drop=True)
    del all_test['label']
    all_train_positive.head()


# In[8]:


if __name__ == "__main__":
    all_train_positive.shape


# In[9]:


if __name__ == "__main__":
    all_train_negative.shape


# In[10]:


if __name__ == "__main__":
    k = 12    # 构建k个数据集
    frac = 5    # 负样本数量 / 正样本数量
    train_list = []
    for i in tqdm(range(k)):
        train_negative = all_train_negative.sample(n=all_train_positive.shape[0] * frac)

        train_all = all_train_positive.append(train_negative)
        train_all = train_all.sample(frac=1).reset_index(drop=True)
        train_list.append(train_all)

    train_all.head()


# ### 开始排序, 构建

# In[12]:


class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32"):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

    def __hash__(self):
        return self.name.__hash__()


# In[13]:


def build_input_features(feature_columns, prefix=''):
    input_features = OrderedDict()
    for fc in feature_columns:
        input_features[fc.name] = Input(
            shape=(fc.dimension,), name=prefix + fc.name, dtype=tf.float32)
    return input_features


# In[14]:


def get_dense_input(features, feature_columns):
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if feature_columns else []
    dense_input_list = []
    for fc in dense_feature_columns:
        dense_input_list.append(features[fc.name])
    return dense_input_list


# ### wide侧

# In[15]:


def get_linear_logit(features_inputs, feature_columns,  use_bias=False, seed=1024, prefix='linear',
                     l2_reg=0):
    dense_input = get_dense_input(features_inputs, feature_columns)


    dense_inputs= tf.keras.layers.Concatenate(axis=-1)(dense_input)


    linear_features = dense_inputs
    linear_logit = tf.keras.layers.Dense(1,activation=None,use_bias=use_bias,
                                   name=prefix)(linear_features)
    return linear_logit


# ### deep侧

# In[16]:


class DNN(Layer):
    """The Multi Layer Percetron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.

        - **activation**: Activation function to use.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not.

        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        # if len(self.hidden_units) == 0:
        #     raise ValueError("hidden_units is empty")
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(
                                            hidden_units[i], hidden_units[i + 1]),
                                        initializer=glorot_normal(
                                            seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed + i) for i in
                               range(len(self.hidden_units))]

        self.activation_layers = [Activation(self.activation) for _ in range(len(self.hidden_units))]

        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        deep_input = inputs

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(
                deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])
#             fc = Dense(self.hidden_size[i], activation=None, \
#                       kernel_initializer=glorot_normal(seed=self.seed), \
#                       kernel_regularizer=l2(self.l2_reg))(deep_input)
            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)

            fc = self.activation_layers[i](fc)

            fc = self.dropout_layers[i](fc, training=training)
            deep_input = fc

        return deep_input

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate, 'seed': self.seed}
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    


# ### 综合

# In[2]:


def Wide_Deep(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(128,128,64,32),
           l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
           dnn_activation='relu', dnn_use_bn=True, task='binary'):
    """Instantiates the wide&deep Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param fm_group: list, group_name of features that will be used to do feature interactions.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    features_inputs = build_input_features(linear_feature_columns + dnn_feature_columns)
#     print('features_inputs', features_inputs)
    
    inputs_list = list(features_inputs.values())

    # LR
    linear_logit = get_linear_logit(features_inputs, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    # DNN
    dense_input = get_dense_input(features_inputs, dnn_feature_columns)
    dense_input = Concatenate(axis=1)(dense_input)
    dnn_input = dense_input
    print('dnn_input',dnn_input)
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                     dnn_use_bn, seed)(dnn_input)
    dnn_logit = Dense(1,activation='sigmoid',name='dnn_logit',
                       kernel_regularizer=tf.keras.regularizers.l2(l2_reg_dnn),
                       use_bias=False)(dnn_output)


    # concat and Activation
    out_put =Add()([linear_logit,dnn_logit])
    out_put = Activation('sigmoid')(out_put)
    print('start model')
    model = tf.keras.models.Model(inputs=inputs_list, outputs=out_put)
    
    return model


# ### 开始对数据集处理

# In[1]:


if __name__ == "__main__":
    dense_features = train_list[0].columns
    fixlen_feature_columns = [DenseFeat(feat, 1,) for feat in dense_features]
    fixlen_feature_columns.pop()
    # print('fixlen_feature_columns', '\n', fixlen_feature_columns)

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = list(dense_features)[:-1]
    for i, data in tqdm(enumerate(train_list)):
        label = data['label'].values
        target = ['label']

    #     feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    #     print("feature_names",feature_names)

        train, test = train_test_split(data, test_size=0.1, random_state=2020)
        train_model_input = {name:train[name].values for name in feature_names}
        test_model_input = {name:test[name].values for name in feature_names}

        # 4.Define Model,train,predict and evaluate
        if i == 0:
            model = Wide_Deep(linear_feature_columns, dnn_feature_columns, task='binary')
            model.compile("adam", "binary_crossentropy",
                          metrics=['binary_crossentropy'], )

    #     print(train_model_input)
    #     print('---')
    #     print(train[target].values)

        history = model.fit(train_model_input, train[target].values,
                            batch_size=128, epochs=10, verbose=2, validation_data=[test_model_input,test[target].values])

        pred_ans = model.predict(test_model_input, batch_size=256)
        print('test accuracy: ', round(accuracy_score(test[target].values, [1 if x > 0.5 else 0 for x in pred_ans.flatten()]), 4))
        print("test LogLoss: ", round(log_loss(test[target].values, pred_ans), 4))
        print("test AUC: ", round(roc_auc_score(test[target].values, pred_ans), 4))
        print()
        
    model.save('./wide_and_deep.h5')

