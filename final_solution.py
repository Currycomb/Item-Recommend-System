#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
# @Author  : Currycomb1


# ## 导入工具包

# In[199]:


import numpy as np
import pickle
import time

import sys
sys.path.append(".")    # 模块父目录下的model文件中，相对路径
print(sys.path)

import tensorflow as tf
from gensim.models import Word2Vec, KeyedVectors

import pandas as pd
from tqdm import tqdm
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

from final_rank import *
from recall_function import *

# ## 开始处理数据集

# ### 读取数据

# In[9]:


phase = 0
nrows = None
recall_num = 500
top_k = 50
train_path = '../data/underexpose_train'
test_path = '../data/underexpose_test'


# In[10]:


# 读取user特征
user_feat = pd.read_csv(
                        train_path + '/underexpose_user_feat.csv'
                        ,header=None
                        ,nrows=nrows
                        ,names=['user_id', 'user_age_level', 'user_gender', 'user_city_level']
                        ,sep=','
                        ,dtype={'user_id':np.str,'user_age_level':np.str,'user_gender':np.str, 'user_city_level':np.str}
                        )


# In[11]:


# 读取item特征
item_feat = pd.read_csv(
                        train_path + '/underexpose_item_feat.csv'
                        ,header=None
                        ,nrows=nrows
                        ,names=['item_id'] + ['txt_vec_'+str(i) for i in range(1, 129)] + ['img_vec_'+str(i) for i in range(1, 129)]
                        ,sep=','
                        )


# In[12]:


user_feat.head()


# In[13]:


item_feat['txt_vec_1'] = item_feat['txt_vec_1'].map(lambda x :float(x.split('[')[1].strip()))
item_feat['img_vec_1'] = item_feat['img_vec_1'].map(lambda x :float(x.split('[')[1].strip()))
item_feat['txt_vec_128'] = item_feat['txt_vec_128'].map(lambda x :float(x.split(']')[0].strip()))
item_feat['img_vec_128'] = item_feat['img_vec_128'].map(lambda x :float(x.split(']')[0].strip()))

feature = item_feat.iloc[:, 1:].values


# ### item_feature 256维太大, PCA降维

# In[14]:


from sklearn.decomposition import PCA
pca = PCA(n_components=50)
feature = pca.fit_transform(feature)


# In[15]:


sum(pca.explained_variance_ratio_)


# In[16]:


item_feat['item_id'] = item_feat['item_id'].map(str)
for i in range(50):
    item_feat['f_'+str(i)] = feature[:, i]
for i in range(1, 129):
    del item_feat['txt_vec_'+str(i)]
    del item_feat['img_vec_'+str(i)]


# In[17]:


item_feat.head()


# In[18]:


length = np.sqrt(np.sum(np.square(feature), axis=1)).reshape(-1, 1)
length = np.tile(length, (1, 50))
temp = feature / length
print(np.sum(np.square(temp)[0, :]))    # 归一化成立


# In[19]:


# 训练集
click_train = pd.read_csv(
                        train_path + '/underexpose_train_click-{phase}.csv'.format(phase=phase)
                        ,header=None
                        ,nrows=nrows
                        ,names=['user_id', 'item_id', 'time']
                        ,sep=','
                        ,dtype={'user_id':np.str,'item_id':np.str,'time':np.str}
                        ) 


# In[20]:


# 测试集
click_test = pd.read_csv(
                        test_path + '/underexpose_test_click-{phase}.csv'.format(phase=phase)
                        ,header=None
                        ,nrows=nrows
                        ,names=['user_id', 'item_id', 'time']
                        ,sep=','
                        ,dtype={'user_id':np.str,'item_id':np.str,'time':np.str}
                        )


# ### 合并数据集

# In[21]:


click_all = click_train.append(click_test)


# In[22]:


click_all.head()    # user_id  item_id  timestamp


# ### 数据集按时间排序

# In[23]:


click_all = click_all.sort_values('time')


# ### 数据按 user_id, item_id, time 去重，保存最后一条

# In[24]:


click_all = click_all.drop_duplicates(['user_id','item_id','time'],keep='last').reset_index(drop=True)


# In[25]:


click_all.shape


# ### 获取用户 训练用户集 和 测试用户集

# In[26]:


set_pred = set(click_test['user_id'])
set_train = set(click_all['user_id']) - set_pred


# ### 获取 训练集合最后一条数据 当作召回的标签评判召回的效果

# In[27]:


# 思路：把训练集用户购买的最后一项商品移除，作为label，来判断是否命中
# 即训练集上用户A购买了10个商品，那么把最后一个购买的商品拎出来作为预测，来判断准确率
# temp_就是为了暂时储存用户购买的最后一个商品
# 其中dict_label_user_item就是训练集对应user_id购买的最后一个item的item_id,所以叫做dict_label

temp_ = click_all
temp_['pred'] = temp_['user_id'].map(lambda x: 'test' if x in set_pred else 'train')
temp_ = temp_[temp_['pred']=='train'].drop_duplicates(['user_id'],keep='last')
temp_['remove'] = 'remove'

train_test = click_all
train_test = train_test.merge(temp_,on=['user_id','item_id','time','pred'],how='left')
train_test = train_test[train_test['remove']!='remove']
    
dict_label_user_item = dict(zip(temp_['user_id'],temp_['item_id']))


# ### 获取热门商品列表，用于补全热度

# In[28]:


temp_ = train_test.groupby(['item_id'])['user_id'].count().reset_index()    # 购买对应item_id的有多少人
temp_ = temp_.sort_values(['user_id'])    # 按照购买人数排列
hot_list = list(temp_['item_id'])[::-1]    # 逆序排列，即hot_list越靠前的商品热度越高
hot_list[:10]


# ### 统计商品频率

# In[29]:


stat_cnt = Counter(list(click_all['item_id'])) 
stat_cnt.most_common(10)


# ### 聚合数据以便后续使用, 一个user_id购买了多个item_id

# In[30]:


group_by_col, agg_col = 'user_id', 'item_id'
data_ = click_all.groupby(['user_id'])[['item_id','time']].agg({'item_id':lambda x:','.join(list(x)), 'time':lambda x:','.join(list(x))}).reset_index()
data_.head(5)


# In[31]:


data_.shape


# ### 平均一个用户购买了几个商品

# In[32]:


stat_length = np.mean([len(item_txt.split(',')) for item_txt in data_['item_id']])
stat_length


# ## 开始召回

# ### glove召回(放弃)，item2vec召回(放弃)，关联规则召回，item_feature召回

# In[33]:


print('-------- 召回 -------------')
# print('click_all:  ', train_test.head(15))
# print('dict_label:  ', dict_label_user_item)     
    
    

# glove召回 太慢了 放弃
# topk_recall_glove = topk_recall_glove_embedding(
#                                             click_all=train_test
#                                             ,dict_label=dict_label_user_item
#                                             ,k=recall_num
#                                             ,dim=88
#                                             ,epochs=1
#                                             ,learning_rate=0.5
#                                             )

# item2vec召回 召回率太低 放弃                                                
# topk_recall_item2vec = topk_recall_word2vec_embedding(
#                                             click_all=train_test
#                                             ,dict_label=dict_label_user_item
#                                             ,k=recall_num
#                                             ,dim=88
#                                             ,epochs=30
#                                             ,learning_rate=0.5
#                                             )

# 关联规则, 效果较好   
topk_recall_association_rules_result, matrix_association_rules = topk_recall_association_rules(
                                            click_all=train_test
                                            ,dict_label=dict_label_user_item
                                            ,k=300
                                            )

# item_feature召回, 训练较慢, 45min左右
# top100_recall_feature = topk_recall_item_feature(
#                                            click_all=train_test,
#                                            dict_label=dict_label_user_item, 
#                                            item_feat=item_feat, 
#                                            feature_matrix=temp
#                                            )


# ### deepwalk和node2vec召回

# In[34]:


# deepwalk和node2vec模型读取
user_data = '../../data/'
deepwalk_model = KeyedVectors.load_word2vec_format(user_data + 'deepwalk_offline.bin', binary=True)
node2vec_model = KeyedVectors.load_word2vec_format(user_data + 'node2vec_offline.bin', binary=True)


# In[35]:


topk_recall_deepwalk_result = pickle.load(open('./top300_recall_deepwalk_result.pkl', 'rb'))
topk_recall_node2vec_result = pickle.load(open('./top300_recall_node2vec_result.pkl', 'rb'))


# ### deepwalk 召回

# In[37]:


# deepwalk召回
# topk_recall_deepwalk_result = topk_recall_deepwalk(
#                             click_all=train_test, 
#                             dict_label=dict_label_user_item, 
#                             deepwalk_model=deepwalk_model, 
#                             k=300
#                             )


# ### node2vec召回

# In[38]:


# node2vec召回
# topk_recall_node2vec_result = topk_recall_node2vec(
#                             click_all=train_test, 
#                             dict_label=dict_label_user_item, 
#                             node2vec_model=node2vec_model, 
#                             k=300
#                             )


# ### 保存deepwalk和node2vec召回结果

# In[ ]:


# 保存数据
# pickle.dump(topk_recall_deepwalk_result, open('./top300_recall_deepwalk_result.pkl', 'wb'))
# pickle.dump(topk_recall_node2vec_result, open('./top300_recall_node2vec_result.pkl', 'wb'))


# ### 尝试基于deepwalk和node2vec增加feature，发现时间太长，遂放弃

# In[ ]:


# 训练开销太大
# item_sim = {}
# import time
# a=time.time()
# for i, item_i in tqdm(enumerate(items_recall)):
#     if item_i not in item_sim:
#         item_sim[item_i] = {}
#     for item_j in items_recall:
#         if item_i in deepwalk_model and item_i in node2vec_model and item_j in deepwalk_model and item_j in node2vec_model:
#             item_sim[item_i][item_j] = [deepwalk_model.wv.similarity(item_i, item_j), node2vec_model.wv.similarity(item_i, item_j)]
#     if i >= 3:
#         print((time.time() - a) / 60, 'min')
#         break


# ### 读取之前保存的item_feature召回结果

# In[40]:


top100_recall_feature = pickle.load(open('./top100_recall_feature.pkl', 'rb'))


# In[41]:


# 保存数据
# pickle.dump(top100_recall_feature, open('./top100_recall_feature.pkl', 'wb'))


# ## 评测召回效果

# In[42]:


def metrics_recall(topk_recall, hot_list, phase, k, sep=10, flag=True):
    # 筛选出rare的商品
    len_hot_list = len(hot_list)
    rare_list = hot_list[len_hot_list // 2: ]
    
    # 筛选出训练集, 并且根据训练集根据'user_id','score_similar'进行排序
    if flag:
        data_ = topk_recall[topk_recall['pred']=='train'].sort_values(['user_id','score_similar'],ascending=False)
    else:
        data_ = topk_recall[topk_recall['pred']=='train'].sort_values(['user_id'],ascending=False)
        
    data_ = data_.groupby(['user_id']).agg({'item_similar':lambda x:list(x),'next_item_id':lambda x:''.join(set(x))})
    
    # index的值代表召回成功的商品位于召回列表的第几位
    data_['index'] = [recall_.index(label_) if label_ in recall_ else -1 for (label_, recall_) in zip(data_['next_item_id'],data_['item_similar'])]
#     print(data_.head(10))
    
    print('-------- 召回效果 -------------')
    print('-------- phase: ', phase,' -------------')
    data_num = len(data_)
    
    for topk in range(sep,k+1,sep):
        
        hit_num_rate_full = 0.0
        hit_num_rate_half = 0.0
        ndcg_k_full = 0.0
        ndcg_k_rare = 0.0
        
        for i, row in data_.iterrows():
        
            if row['index'] != -1:
                if row['index'] <= topk:
                    ndcg_k_full += 1 / np.log2(row['index'] + 2)
                    hit_num_rate_full += 1
                if row['index'] <= topk and row['next_item_id'] in rare_list:
                    ndcg_k_rare += 1 / np.log2(row['index'] + 2)
                    hit_num_rate_half += 1

        hit_num_rate_full /= data_num
        hit_num_rate_half /= data_num
        
        print('phase: ', phase, ' top_', topk, ' : ', 'hit_num_rate_full : ', round(hit_num_rate_full, 4), 'hit_num_rate_half : ', round(hit_num_rate_half, 4), ' data_num : ', data_num)
#         print('ndcg_{k}_full: {ndcg_k_full}    '.format(k=topk, ndcg_k_full=ndcg_k_full), 'ndcg_{k}_rare: {ndcg_k_rare}'.format(k=topk, ndcg_k_rare=ndcg_k_rare)) 
        print()
        
    return [hit_num_rate_full, hit_num_rate_half]


# In[43]:


print('-------- 评测召回效果 -------------')
recall_num = 300
# hit_rate_glove = metrics_recall(topk_recall=topk_recall_glove, hot_list=hot_list, phase=phase, k=recall_num, sep=int(recall_num/5))
# print('-------- glove召回TOP:{k}时, full命中百分比:{hit_num_rate_full}, rare命中百分比:{hit_num_rate_half} -------------'.format(k=recall_num, hit_num_rate_full=round(hit_rate_glove[0], 4), hit_num_rate_half=round(hit_rate_glove[1], 4)))
# print()

# hit_rate_item2vec = metrics_recall(topk_recall=topk_recall_item2vec, hot_list=hot_list, phase=phase, k=recall_num, sep=int(recall_num/5))
# print('-------- item2vec召回TOP:{k}时, full命中百分比:{hit_num_rate_full}, rare命中百分比:{hit_num_rate_half} -------------'.format(k=recall_num, hit_num_rate_full=round(hit_rate_item2vec[0], 4), hit_num_rate_half=round(hit_rate_item2vec[1], 4)))
# print()

hit_rate_deepwalk = metrics_recall(topk_recall=topk_recall_deepwalk_result, hot_list=hot_list, phase=phase, k=recall_num, sep=int(recall_num/5))
print('-------- deepwalk召回TOP:{k}时, full命中百分比:{hit_num_rate_full}, rare命中百分比:{hit_num_rate_half} -------------'.format(k=recall_num, hit_num_rate_full=round(hit_rate_deepwalk[0], 4), hit_num_rate_half=round(hit_rate_deepwalk[1], 4)))
print()

hit_rate_node2walk = metrics_recall(topk_recall=topk_recall_node2vec_result, hot_list=hot_list, phase=phase, k=recall_num, sep=int(recall_num/5))
print('-------- node2vec召回TOP:{k}时, full命中百分比:{hit_num_rate_full}, rare命中百分比:{hit_num_rate_half} -------------'.format(k=recall_num, hit_num_rate_full=round(hit_rate_node2walk[0], 4), hit_num_rate_half=round(hit_rate_node2walk[1], 4)))
print()

hit_rate_association = metrics_recall(topk_recall=topk_recall_association_rules_result, hot_list=hot_list, phase=phase, k=recall_num, sep=int(recall_num/5))
print('-------- 关联规则召回TOP:{k}时, full命中百分比:{hit_num_rate_full}, rare命中百分比:{hit_num_rate_half} -------------'.format(k=recall_num, hit_num_rate_full=round(hit_rate_association[0], 4), hit_num_rate_half=round(hit_rate_association[1], 4)))
print()

hit_rate_feature = metrics_recall(topk_recall=top100_recall_feature, hot_list=hot_list, phase=phase, k=100, sep=25, flag=False)
print('-------- items_feature召回TOP:100时, full命中百分比:{hit_num_rate_full}, rare命中百分比:{hit_num_rate_half} -------------'.format(hit_num_rate_full=round(hit_rate_feature[0], 4), hit_num_rate_half=round(hit_rate_feature[1], 4)))


# ## 多路召回：关联规则召回300个, item_feature召回100个, deepwalk召回300个, node2vec召回300个

# In[44]:


top100_recall_feature.head()


# In[45]:


# 利用log降低数据方差
topk_recall_association_rules_result['score_similar'] = topk_recall_association_rules_result['score_similar'].map(lambda x: np.log(10 + x * 120) - 2 if x > 0 else x)


# In[112]:


top100_recall_feature


# In[116]:


v = topk_recall[topk_recall['user_id'] == '1']
v[v['pred'] == 'train']


# In[46]:


topk_recall = pd.concat([topk_recall_association_rules_result, topk_recall_deepwalk_result, topk_recall_node2vec_result, top100_recall_feature], axis=0)
topk_recall = topk_recall.drop_duplicates(subset=['user_id', 'item_similar'])    # 18,505,000 --> 14,166,021
topk_recall.shape


# In[47]:


# 为item_feature召回填充分数
topk_recall = topk_recall.fillna(0)


# In[48]:


hit_rate_all = metrics_recall(topk_recall=topk_recall, hot_list=hot_list, phase=phase, k=1000, sep=200, flag=False)
print('-------- 召回TOP:1000时, full命中百分比:{hit_num_rate_full}, rare命中百分比:{hit_num_rate_half} -------------'.format(hit_num_rate_full=round(hit_rate_all[0], 4), hit_num_rate_half=round(hit_rate_all[1], 4)))


# In[49]:


topk_recall.head()


# ## 获取label, 召回的item == item_next 真实, 则为1。
# ## 为了减少训练样本，降低内存的压力, 只取召回样本中存在真实next_item_id的user进行训练

# In[50]:


data_list = []

print('------- 构建样本 -----------')
temp_ = topk_recall
temp_['label'] = [ 1 if next_item_id == item_similar else 0 for (next_item_id, item_similar) in zip(temp_['next_item_id'], temp_['item_similar'])]

set_user_label_1 = set(temp_[temp_['label']==1]['user_id'])
temp_['keep'] = temp_['user_id'].map(lambda x: 1 if x in set_user_label_1 else 0)
train_data = temp_[temp_['keep']==1][['user_id','item_similar','score_similar','label']]

# temp_['pred'] = temp_['user_id'].map(lambda x: 'test' if x in set_pred else 'train')
test_data = temp_[temp_['pred']=='test'][['user_id','item_similar','score_similar']]

train_data.head()


# ## 加入用户行为序列 方便后续构建特征

# In[51]:


# 里面包含最近购买的item,这是个坑
train_data = train_data.merge(data_,on=['user_id'],how='left')
test_data = test_data.merge(data_,on=['user_id'],how='left')
list_train_test = [('train', train_data), ('test', test_data)]


# In[52]:


train_data.head()


# In[53]:


test_data.head()


# In[55]:


print('train_data rows: ', train_data.shape[0])
print('test_data rows: ', test_data.shape[0])


# ### --------------------------------------   run all above   --------------------------------------

# ## 加入训练特征，为排序做准备

# In[ ]:


data_list = []
for flag, data in list_train_test:

    print('------- 加入特征 {flag} -----------'.format(flag=flag))
        
    list_train_flag, list_user_id, list_item_similar, list_label, list_features = [], [], [], [], []
    
    for i,row in tqdm(data.iterrows()):    # 对df的每一行进行迭代

        user_id, item_id, score_similar = str(row['user_id']), str(row['item_similar']), float(row['score_similar'])
        
        list_item_id = row['item_id'].split(',')[::-1][1:]    # 逆序排列,list_item_id[0]即为最近期购买的一个item,
        
        # stat_cnt是每个商品被购买次数的字典
        feature = [score_similar, len(list_item_id), stat_cnt[item_id]]
        feature_col_name = ['score_similar','len_item_clicked','recall_item_cnt']
        
        len_ = len(list_item_id)
        
        for i in range(6):
            if i < len_:
                item_i = list_item_id[i]
                
                # item_i是实际购买的商品, item_id是召回的商品
                feature += [stat_cnt[item_i]]
                
                if (item_i in node2vec_model) and (item_id in node2vec_model):
                    feature += [node2vec_model.wv.similarity(item_i, item_id)]
                else:
                    feature += [0]
                
                if (item_i in deepwalk_model) and (item_id in deepwalk_model):
                    feature += [deepwalk_model.wv.similarity(item_i, item_id)]
                else:
                    feature += [0]
                
                if (item_i in matrix_association_rules) and (item_id in matrix_association_rules[item_i]):
                    feature += [matrix_association_rules[item_i][item_id]]
                else:
                    feature += [0]
                    
                if (item_id in matrix_association_rules) and (item_i in matrix_association_rules[item_id]):
                    feature += [matrix_association_rules[item_id][item_i]]
                else:
                    feature += [0]
        
            else:
                feature += [0] * 5
                
            feature_col_name += ['clicked_item_'+str(i)+'_cnt',
                                 'node2vec_item_'+str(i)+'_to_recall_item_'+str(i)+'_score',
                                 'deepwalk_item_'+str(i)+'_to_recall_item_'+str(i)+'_score'
                                 'clicked_item_'+str(i)+'_to_recall_item_'+str(i)+'_score',
                                 'recall_item_'+str(i)+'_to_'+'clicked_item_'+str(i)+'_score']
        
        
        list_features.append(feature)
        list_train_flag.append(flag)
        list_user_id.append(user_id)
        list_item_similar.append(item_id)

        if flag == 'train':
            label = int(row['label'])
            list_label.append(label)

        if flag == 'test':  
            label = -1
            list_label.append(label)

    feature_all = pd.DataFrame(list_features)
    feature_all.columns = ['f_'+str(i) for i in range(len(feature_all.columns))]

    feature_all['train_flag'] = list_train_flag
    feature_all['user_id'] = list_user_id
    feature_all['item_similar'] = list_item_similar
    feature_all['label'] = list_label

    data_list.append(feature_all)

feature_all_train_test = pd.concat(data_list)


print('--------------------------- 特征数据 ---------------------')
len_f = len(feature_all_train_test)
len_train = len(feature_all_train_test[feature_all_train_test['train_flag']=='train'])
len_test = len(feature_all_train_test[feature_all_train_test['train_flag']=='test'])
len_train_1 = len(feature_all_train_test[(feature_all_train_test['train_flag']=='train') & (feature_all_train_test['label']== 1)]) 
print('所有数据条数', len_f)
print('训练数据 : ', len_train)
print('训练数据 label 1 : ', len_train_1)
print('训练数据 1 / 0 rate : ', len_train_1 * 1.0 / len_f)
print('测试数据 : ' , len_test)
print('flag : ', set(feature_all_train_test['train_flag']))
print('--------------------------- 特征数据 ---------------------')


# In[56]:


# pickle.dump(feature_all_train_test, open('./feature_all_train_test.pkl', 'wb'))
with open('./feature_all_train_test.pkl', 'rb') as f:
    feature_all_train_test = pickle.load(f)


# In[57]:


feature_all_train_test.head()


# In[58]:


feature_all_train_test.shape


# # 排序

# ## 导入模型，进行rank排序序列。训练的x特征就是user历史购买过的商品与召回的商品之间的相似度以及交互特征
# ## label y是召回的item是否hit到

# In[59]:


wide_deep_model_all = tf.keras.models.load_model('./wide_and_deep.h5', custom_objects={'DNN': DNN})
wide_deep_model_no_cv = tf.keras.models.load_model('./wide_and_deep_no_cv.h5', custom_objects={'DNN': DNN})


# In[60]:


all_test = feature_all_train_test[feature_all_train_test['train_flag'] == 'test'].drop('train_flag', axis=1)
all_train = feature_all_train_test[feature_all_train_test['train_flag'] == 'train'].drop('train_flag', axis=1)
del feature_all_train_test


# In[61]:


all_train.head()


# In[62]:


all_train_positive = all_train[all_train['label'] == 1].drop(['item_similar', 'user_id'], axis=1).reset_index(drop=True)
all_train_negative = all_train[all_train['label'] == 0].drop(['item_similar', 'user_id'], axis=1).reset_index(drop=True)
del all_test['label']
all_train_positive.head()


# ### 构建验证集

# In[129]:


valid = 0.2 

# for df in [all_train_positive, all_train_negative]:
#     cut_idx = int(round(valid * df.shape[0]))
#     df_train_no_cv, df_cv = df.iloc[:, cut_idx:], df.iloc[:, :cut_idx]
    
cut_idx = int(round(valid * all_train.shape[0]))
df_train_no_cv, df_cv = all_train.drop(['item_similar', 'user_id'], axis=1).iloc[cut_idx:, :], all_train.drop(['item_similar', 'user_id'], axis=1).iloc[:cut_idx, :]

print('训练集维度: ', df_train_no_cv.shape)
print('验证集维度: ', df_cv.shape)


# In[125]:


all_train = all_train.sort_values(by='user_id').reset_index(drop=True)


# In[126]:


all_train[all_train['user_id'] == '1']


# In[127]:


cut_idx


# In[128]:


df_user_item_prob_cv = all_train[['item_similar', 'user_id', 'label']].iloc[:cut_idx, :]
df_user_item_prob_cv = df_user_item_prob_cv.loc[:, ['user_id', 'item_similar', 'label']]

assert df_user_item_prob_cv.shape[0] == df_cv.shape[0]
df_user_item_prob_cv.head()


# In[130]:


df_cv.head()


# In[131]:


df_user_item_prob_cv.shape


# In[132]:


df_cv.shape


# In[96]:


train_no_cv_positive = df_train_no_cv[df_train_no_cv['label'] == 1].reset_index(drop=True)
train_no_cv_negative = df_train_no_cv[df_train_no_cv['label'] == 0].reset_index(drop=True)

# train_cv_positive = df_cv[df_cv['label'] == 1].reset_index(drop=True)
# train_cv_negative = df_cv[df_cv['label'] == 0].reset_index(drop=True)


# In[168]:


train_no_cv_list = build_train_list(train_no_cv_positive, train_no_cv_negative)
train_all_list = build_train_list(all_train_positive, all_train_negative)


# In[94]:


def build_train_list(train_positive, train_negative, k=12, frac=5):
    # 构建k个数据集
    # 负样本数量 / 正样本数量
    train_list = []
    for i in tqdm(range(k)):
        train_negative_part = train_negative.sample(n=all_train_positive.shape[0] * frac)

        train_all = train_positive.append(train_negative_part)
        train_all = train_all.sample(frac=1).reset_index(drop=True)
        train_list.append(train_all)

#     train_all.head()
    return train_list


# In[103]:


def train_model_widedeep(train_list):
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
        
        train_model_input = {name:data[name].values for name in feature_names}

        if i == 0:
            model = Wide_Deep(linear_feature_columns, dnn_feature_columns, task='binary')
            model.compile("adam", "binary_crossentropy",
                          metrics=['binary_crossentropy'], )

    #     print(train_model_input)
    #     print('---')
    #     print(train[target].values)

        history = model.fit(train_model_input, data[target].values,
                            batch_size=128, epochs=10, verbose=2)

    #     pred_ans = model.predict(test_model_input, batch_size=256)
    #     print('test accuracy: ', round(accuracy_score(test[target].values, [1 if x > 0.5 else 0 for x in pred_ans.flatten()]), 4))
    #     print("test LogLoss: ", round(log_loss(test[target].values, pred_ans), 4))
    #     print("test AUC: ", round(roc_auc_score(test[target].values, pred_ans), 4))
    #     print()
    
    return model


# In[104]:


model_no_cv = train_model_widedeep(train_no_cv_list)
# model_all = train_model_widedeep(train_all_list)

model_no_cv.save('./wide_and_deep_no_cv.h5')
# model_all.save('./wide_and_deep_all.h5')


# ### 获取验证集上预测的label

# In[133]:


df_cv.head()


# In[134]:


cv_model_input = {name: df_cv[name].values for name in df_cv.columns[:-1]}
pred_prob_cv = wide_deep_model_no_cv.predict(cv_model_input, batch_size=256) 


# In[135]:


df_user_item_prob_cv['pred_prob'] = pred_prob_cv
df_user_item_prob_cv.head()


# ### 验证集处理

# In[136]:


df_user_item_prob_cv['f_0'] = df_cv['f_0']


# In[137]:


df_user_item_prob_cv.head(10)


# In[141]:


sum(df_user_item_prob_cv.label)


# In[142]:


len(set(df_user_item_prob_cv.user_id))


# In[140]:


# with open('./df_user_item_prob_cv.pkl', 'wb') as f_:
#     pickle.dump(df_user_item_prob_cv, f_)

# with open('./topk_recall.pkl', 'wb') as f_:    
#     pickle.dump(topk_recall, f_)


# In[82]:


df_user_item_prob_cv = pickle.load(open('./df_user_item_prob_cv.pkl', 'rb'))
topk_recall = pickle.load(open('./topk_recall.pkl', 'rb'))


# In[108]:


train_eval[train_eval['user_id'] == '2']


# ## 验证集验证效果, 将召回1000个并经过wide&deep排序后得到的前50个的结果和直接召回1000个选
# 
# ## 其中相似度最高的50个不训练模型的结果进行对比

# In[167]:


train_eval_rank = pickle.load(open('./train_eval_rank.pkl', 'rb'))


# In[151]:


train_eval = df_user_item_prob_cv
len_hot = len(hot_list)
high_half_item, low_half_item = hot_list[:len_hot//2], hot_list[len_hot//2:]

a = time.time()
print('---------- 准备添加half列 ----------')
train_eval['half'] = train_eval['item_similar'].map(lambda x: 1 if x in low_half_item else 0)
print('---------- 添加了half列 ----------')
print('time: ', round((time.time() - a) / 60, 4), 'mins')

a = time.time()
topk = 50
# f_0, f_1, f_2 特征分别是'score_similar','len_item_clicked','recall_item_cnt'
train_eval['rank'] = train_eval.groupby(['user_id'])['pred_prob'].rank(ascending=False, method='first')
print('---------- 添加了rank列 ----------')
print('time: ', round((time.time() - a) / 60, 4), 'mins')

a = time.time()
train_eval['rank_recall'] = train_eval.groupby(['user_id'])['f_0'].rank(ascending=False, method='first')
print('---------- 添加了rank_recall列 ----------')
print('time: ', round((time.time() - a) / 60, 4), 'mins')

train_eval_rank = train_eval[train_eval['rank']<=topk]
train_eval_rank_recall = train_eval[train_eval['rank_recall']<=topk]


# In[144]:


df_cv.head()


# In[145]:


train_eval_rank = train_eval_rank.reset_index(drop=True)
train_eval_rank.head()


# In[166]:


# 对分数做惩罚，取1 / log(stat_cnt[item_i] + 1)
train_eval_rank = train_eval_rank['pred_prob'].map(lambda x: x / np.log10(stat_cnt[x] + 1))


# In[152]:


# with open('./train_eval_rank.pkl', 'wb') as f_:
#     pickle.dump(train_eval_rank, f_)


# In[147]:


recall_rate_full, recall_rate_half = hit_rate_all[0], hit_rate_all[1]
recall_rate_full


# In[149]:


np.sum(train_eval_rank['label'])


# In[153]:


# 别忘记乘recall_rate
len_user_id = len(set(train_eval.user_id))
hitrate_50_full = np.sum(train_eval_rank['label']) / len_user_id * recall_rate_full
hitrate_50_half = np.sum(train_eval_rank['label'] * train_eval_rank['half']) / len_user_id * recall_rate_half
ndcg_50_full = np.sum(train_eval_rank['label'] / np.log2(train_eval_rank['rank'] + 2.0) * recall_rate_full)
ndcg_50_half = np.sum(train_eval_rank['label'] * train_eval_rank['half'] / np.log2(train_eval_rank['rank'] + 2.0) * recall_rate_half)

print("------------- eval wide&deep result -------------")
print("hitrate_50_full : ", hitrate_50_full, 'ndcg_50_full : ', ndcg_50_full, '\n')
print("hitrate_50_half : ", hitrate_50_half, 'ndcg_50_half : ', ndcg_50_half, '\n')
print("------------- eval wide&deep result -------------")


# In[154]:


hitrate_50_full = np.sum(train_eval_rank_recall['label']) / len_user_id * recall_rate_full
hitrate_50_half = np.sum(train_eval_rank_recall['label'] * train_eval_rank_recall['half']) / len_user_id * recall_rate_half
ndcg_50_full = np.sum(train_eval_rank_recall['label'] / np.log2(train_eval_rank_recall['rank_recall'] + 2.0) * recall_rate_full)
ndcg_50_half = np.sum(train_eval_rank_recall['label'] * train_eval_rank_recall['half'] / np.log2(train_eval_rank_recall['rank_recall'] + 2.0) * recall_rate_half)

print("------------- eval origin result -------------")
print("hitrate_50_full : ", hitrate_50_full, 'ndcg_50_full : ', ndcg_50_full, '\n')
print("hitrate_50_half : ", hitrate_50_half, 'ndcg_50_half : ', ndcg_50_half, '\n')
print("------------- eval origin result -------------")


# ## 进行预测

# In[187]:


all_test.head()


# In[171]:


test_model_input = {name: all_test[name].values for name in all_test.columns[:-2]}
pred_prob = wide_deep_model_all.predict(test_model_input, batch_size=256)


# In[172]:


all_test['pred_prob'] = pred_prob
all_test = all_test[['user_id', 'item_similar', 'pred_prob']]
all_test['pred_prob'] = all_test['pred_prob'].map(lambda x: x / np.log10(stat_cnt[x] + 1))
all_test.head()


# In[183]:


all_test['rank'] = all_test.groupby(['user_id'])['pred_prob'].rank(ascending=False, method='first')


# In[188]:


all_test = all_test[all_test['rank']<=50].reset_index(drop=True)
all_test = all_test.sort_values(['rank'])
submit = all_test.groupby(['user_id'])['item_similar'].agg(lambda x:','.join(list(x))).reset_index()


# In[189]:


submit.head()


# In[191]:


for i,row in submit.iterrows():
    txt_item = row['item_similar'].split(',')
    assert len(txt_item) == topk, '推荐的商品不为{}个'.format(topk)


# ## 保存预测结果

# In[192]:


def save(submit_all,topk):
    time_str = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    file_name = './result{time_str}.csv'.format(time_str=time_str)
    with open(file_name, 'w') as f:
        for i, row in submit_all.iterrows():
            
            user_id = str(row['user_id'])
            item_list = str(row['item_similar']).split(',')[:topk]
            assert len(set(item_list)) == topk
            
            line = user_id + ',' + ','.join(item_list) + '\n'
            assert len(line.strip().split(',')) == (topk+1)
            
            f.write(line)


# In[193]:


save(submit, 50)

