#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
from gensim.models import Word2Vec, KeyedVectors
import pandas as pd
from tqdm import tqdm
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# glove embedding 召回
def topk_recall_glove_embedding(
                                click_all
                                ,dict_label
                                ,k=100
                                ,dim=88
                                ,epochs=30
                                ,learning_rate=0.5
                                ):
    
    import psutil
    from glove import Glove
    from glove import Corpus
    
    data_ = click_all.groupby(['pred','user_id'])['item_id'].agg(lambda x:','.join(list(x))).reset_index()
    list_data = list(data_['item_id'].map(lambda x:x.split(',')))

    corpus_model = Corpus()
    corpus_model.fit(list_data, window=999999)
    
    glove = Glove(no_components=dim, learning_rate=learning_rate)
    glove.fit(corpus_model.matrix, epochs=epochs, no_threads=psutil.cpu_count(), verbose=True)
    glove.add_dictionary(corpus_model.dictionary)

    list_user_id = []
    list_item_similar = []
    list_score_similar = []
    print('------- glove 召回 ---------')
    for i, row in tqdm(data_.iterrows()):
        
        list_item_id = row['item_id'].split(',')

        dict_item_id_score = {}
        for i, item in enumerate(list_item_id[::-1]):
            most_topk = glove.most_similar(item, number=k)
            for item_similar, score_similar in most_topk:
                if item_similar not in list_item_id:
                    if item_similar not in dict_item_id_score:
                        dict_item_id_score[item_similar] = 0
                    sigma = 0.8
                    dict_item_id_score[item_similar] += 1.0 / (1 + sigma * i) * score_similar
        dict_item_id_score_topk = sorted(dict_item_id_score.items(), key=lambda kv: kv[1], reverse=True)[:k]
        assert len(dict_item_id_score_topk) == k
        dict_item_id_set = set([item_similar for item_similar, score_similar in dict_item_id_score_topk])
        assert len(dict_item_id_set) == k
        for item_similar, score_similar in dict_item_id_score_topk:
            list_item_similar.append(item_similar)
            list_score_similar.append(score_similar)
            list_user_id.append(row['user_id'])
            
    
    topk_recall = pd.DataFrame({'user_id':list_user_id,'item_similar':list_item_similar,'score_similar':list_score_similar})
    topk_recall['next_item_id'] = topk_recall['user_id'].map(dict_label)
    topk_recall['pred'] = topk_recall['user_id'].map(lambda x: 'train' if x in dict_label else 'test')

    return topk_recall


# In[3]:


# word2vec embedding召回
def topk_recall_word2vec_embedding(
                                click_all
                                ,dict_label
                                ,k=100
                                ,dim=88
                                ,epochs=30
                                ,learning_rate=0.5
                                ):
    
    import psutil
    import gensim
    
    # 将测试集和训练集的每一个userid购买的商品进行合并
    data_ = click_all.groupby(['pred','user_id'])['item_id'].agg(lambda x:','.join(list(x))).reset_index()
#     print('data_: ', data_.head(10))
    list_data = list(data_['item_id'].map(lambda x:x.split(',')))
#     print('list_data', list_data[:50])
    
    # 建立word2vec模型
    model = gensim.models.Word2Vec(
                    list_data,
                    size=dim,
                    alpha=learning_rate,
                    window=999999,
                    min_count=1,
                    workers=psutil.cpu_count(),
                    compute_loss=True,
                    iter=epochs,
                    hs=0,
                    sg=1,    # skip_gram方法训练
                    seed=42
                )

    
    list_user_id = []
    list_item_similar = []
    list_score_similar = []
    print('------- item2vec 召回 ---------')
    for i, row in tqdm(data_.iterrows()):    # 对dataframe进行行遍历
        
        list_item_id = row['item_id'].split(',')    # 提取出每个用户的item_id

        dict_item_id_score = {}
        for i, item in enumerate(list_item_id[::-1]):
            most_topk = model.wv.most_similar(item, topn=k)    # 提取出对A用户购买的a商品前100个最相似的item
#             print('------most_topk------')
#             print(most_topk)    # 格式是[(item1, score1), (item2, score2), ...]
            
            for item_similar, score_similar in most_topk:
                if item_similar not in list_item_id:
                    
                    if item_similar not in dict_item_id_score:    # 初始化对A用户推荐的商品的字典
                        dict_item_id_score[item_similar] = 0
                        
                    sigma = 0.8
                    dict_item_id_score[item_similar] += 1.0 / (1 + sigma * i) * score_similar
        
        # 根据评分选择前k个(item, score)的键值对
        dict_item_id_score_topk = sorted(dict_item_id_score.items(), key=lambda kv: kv[1], reverse=True)[:k]
        assert len(dict_item_id_score_topk) == k, '召回的商品数目不为{k}个'.format(k=k)
#         print('dict_item_id_score_topk', dict_item_id_score_topk)
        
        # 提取出对应的item_id的集合
        dict_item_id_set = set([item_similar for item_similar, score_similar in dict_item_id_score_topk])
        assert len(dict_item_id_set) == k, '召回的商品数目不为{k}个'.format(k=k)
#         print('dict_item_id_set', dict_item_id_set)
        
        
        for item_similar, score_similar in dict_item_id_score_topk:
            list_item_similar.append(item_similar)
            list_score_similar.append(score_similar)
            list_user_id.append(row['user_id'])
            
    # 创建dataframe
    topk_recall = pd.DataFrame({'user_id':list_user_id,'item_similar':list_item_similar,'score_similar':list_score_similar})
    topk_recall['next_item_id'] = topk_recall['user_id'].map(dict_label)
    topk_recall['pred'] = topk_recall['user_id'].map(lambda x: 'train' if x in dict_label else 'test')

    return topk_recall


# In[4]:


# 关联规则召回
def topk_recall_association_rules(
                                  click_all
                                 ,dict_label
                                 ,k=100
                                 ):
    """
        关联矩阵：按距离加权 
        scores_A_to_B = weight * N_cnt(A and B) / N_cnt(A) => P(B|A) = P(AB) / P(A)
    """
    from collections import Counter    # 引入计数器字典
 
    data_ = click_all.groupby(['user_id'])['item_id'].agg(lambda x:','.join(list(x))).reset_index()
#     print(data_.head(10))
    
    # 按照购买次数生成热度列表, 越在列表前面的被购买次数越多
    hot_list = list(click_all['item_id'].value_counts().index[:].values)
    
    # 生成每个item被购买次数的字典
    stat_cnt = Counter(list(click_all['item_id'])) 
    
    # 每个user购买的item的平均个数
    stat_length = np.mean([ len(item_txt.split(',')) for item_txt in data_['item_id']])
    
    # 初始化关联规则矩阵
    matrix_association_rules = {}
    print('------- association rules matrix 生成 ---------')
    for i, row in tqdm(data_.iterrows()):    # 对每一行进行循环遍历
        
        list_item_id = row['item_id'].split(',')    # 提取出每一个用户购买的item的列表
        len_list_item = len(list_item_id)    # 获取每个用户购买的item的个数
                   

        for i, item_i in enumerate(list_item_id):
            for j, item_j in enumerate(list_item_id):

                if i <= j:
                    if item_i not in matrix_association_rules:
                            matrix_association_rules[item_i] = {}
                    if item_j not in matrix_association_rules[item_i]:
                            matrix_association_rules[item_i][item_j] = 0
                            
                    # 有个问题是这个公式怎么来的? ? ?                    
                    alpha, beta, gama = 1.0, 0.8, 0.8
                    matrix_association_rules[item_i][item_j] += 1.0 * alpha  / (beta + np.abs(i-j)) * 1.0 / stat_cnt[item_i] * 1.0 / (1 + gama * len_list_item / stat_length)
                if i >= j:
                    if item_i not in matrix_association_rules:
                        matrix_association_rules[item_i] = {}
                    if item_j not in matrix_association_rules[item_i]:
                        matrix_association_rules[item_i][item_j] = 0
                    
                    alpha, beta, gama = 0.5, 0.8, 0.8
                    matrix_association_rules[item_i][item_j] += 1.0 * alpha  / (beta + np.abs(i-j)) * 1.0 / stat_cnt[item_i] * 1.0 / (1 + gama * len_list_item / stat_length)
        
        # 可视化关联规则矩阵
#         matrix = []
#         for _, son_dict in matrix_association_rules.items():
#             matrix.append([score for score in son_dict.values()])
#         matrix = np.array(matrix)
#         print(matrix)
#         break
            
    # print(len(matrix_association_rules.keys()))
    # print(len(set(click_all['item_id'])))
    # print('data - matrix: ')
    # print( set(click_all['item_id']) - set(matrix_association_rules.keys()) )
    # print('matrix - data: ')
    # print( set(matrix_association_rules.keys()) - set(click_all['item_id']))
    assert len(matrix_association_rules.keys()) == len(set(click_all['item_id']))

    list_user_id = []
    list_item_similar = []
    list_score_similar = []
    print('------- association rules 召回 ---------')
    
    # 这层循环是对每个用户
    for i, row in tqdm(data_.iterrows()):    # 对数据集每一行进行循环遍历
        
        list_item_id = row['item_id'].split(',')    # 获取每一个用户的购买的item

        dict_item_id_score = {}
        
        # 这层循环是对用户购买过的每个商品依次进行分析，找到每个商品对应的k个商品
        # 换句话说，如果用户A购买了6个商品，那么我们挖掘出6 * k个商品，从6 * k个商品中挑出评分最大的进行召回
        for i, item_i in enumerate(list_item_id[::-1]):
            
            # 这层循环是对用户A购买的某个商品a，和a有关的所有商品进行遍历
            # 对和item_i关联紧密的前k个商品进行遍历, 其中按照关联度降序排列, 商品item_j对应的关联度分数是score_similar
            for item_j, score_similar in sorted(matrix_association_rules[item_i].items(), reverse=True)[0:k]:
                if item_j not in list_item_id:    # item_j不在用户购买的商品里
                    if item_j not in dict_item_id_score:
                        dict_item_id_score[item_j] = 0
                    sigma = 0.8
                    dict_item_id_score[item_j] +=  1.0 / (1 + sigma * i) * score_similar    # 对item_j打分

        dict_item_id_score_topk = sorted(dict_item_id_score.items(), key=lambda kv: kv[1], reverse=True)[:k]    #[(item1, score1), (), ...]
        dict_item_id_set = set([item_similar for item_similar, score_similar in dict_item_id_score_topk])
 
        # 不足的热度补全
        # 召回的商品小于k个，即用户id信息不全导致的冷启动问题
        if len(dict_item_id_score_topk) < k:
            
            # 遍历热门商品, 用热门商品补足
            for i, item in enumerate(hot_list):
                
                # 如果用户没有买过该热门商品并且也没有召回该热门商品
                if (item not in list_item_id) and (item not in dict_item_id_set):
                    item_similar = item
                    score_similar = max(-1, - i * 0.01)    # 将分数设置为负， 即不相关
                    dict_item_id_score_topk.append( (item_similar, score_similar) )
                if len(dict_item_id_score_topk) == k:
                    break

        assert len(dict_item_id_score_topk) == k, '召回的商品不足{}个'.format(k)
        
        # 召回商品id的集合
        dict_item_id_set = set([item_similar for item_similar, score_similar in dict_item_id_score_topk])
        assert len(dict_item_id_set) == k, '召回的商品不足{}个'.format(k)
        
        for item_similar, score_similar in dict_item_id_score_topk:
            list_item_similar.append(item_similar)
            list_score_similar.append(score_similar)
            list_user_id.append(row['user_id'])
            
    topk_recall = pd.DataFrame({'user_id':list_user_id,'item_similar':list_item_similar,'score_similar':list_score_similar})
    topk_recall['next_item_id'] = topk_recall['user_id'].map(dict_label)
    topk_recall['pred'] = topk_recall['user_id'].map(lambda x: 'train' if x in dict_label else 'test')

    return topk_recall, matrix_association_rules


# In[5]:


# 根据物品属性计算物品相似度然后召回
def topk_recall_item_feature(
                                click_all, 
                                dict_label, 
                                item_feat, 
                                feature_matrix, 
                                k=100
                                ):
    data_ = click_all.groupby(['user_id'])['item_id'].agg(lambda x:','.join(list(x))).reset_index()
    print(data_.head())
    print(data_.shape)
    
    all_list, list_user_id, list_item_similar = [], [], []
    for i, row in tqdm(data_.iterrows()):
        list_item_id = row['item_id'].split(',')    # 获取每一个用户的购买的item
        dict_item_id_score = {}
        
        self_items_list = []
        for i, item_i in enumerate(list_item_id[::-1][:10]):    # 选最近购买的10个商品
            
            if item_i in item_feat['item_id'].values:
                first = feature_matrix[item_feat[item_feat['item_id'] == item_i].index[0], :].reshape(1, -1)
#                 print(np.sum(np.square(first)[0, :]))    # 归一化成立
            else:
                continue
                
            item_i_cos_vec = np.dot(first, feature_matrix.T)    
            temp_ = item_i_cos_vec.copy()
            temp_.sort()
            temp_ = temp_[0, :-1].reshape(1, -1)
#             print([i for i in temp_[0, -10:]])
            loc = list(np.argwhere(item_i_cos_vec > temp_[0, -10])[:, 1])    # 对每个商品选10个在属性上最相似的
            self_items_list += list(item_feat.iloc[loc, :]['item_id'])
            
            
        if len(self_items_list) < 100:
            
            # 遍历热门商品, 用热门商品补足
            for i, item in enumerate(hot_list):
                
                # 如果用户没有买过该热门商品并且也没有召回该热门商品
                if (item not in list_item_id) and (item not in self_items_list):
                    self_items_list.append(item)
                if len(self_items_list) == 100:
                    break
        
        assert len(self_items_list) == 100, '召回的商品不为100个'
        
        for _ in range(100):
            list_user_id.append(row['user_id'])
            
        list_item_similar += self_items_list
        
    topk_recall = pd.DataFrame({'user_id':list_user_id,'item_similar':list_item_similar})
    topk_recall['next_item_id'] = topk_recall['user_id'].map(dict_label)
    topk_recall['pred'] = topk_recall['user_id'].map(lambda x: 'train' if x in dict_label else 'test')
        
    return topk_recall


# In[6]:


# deepwalk召回
def topk_recall_deepwalk(
                            click_all, 
                            dict_label, 
                            deepwalk_model, 
                            k=500
                            ):
    data_ = click_all.groupby(['user_id'])['item_id'].agg(lambda x:','.join(list(x))).reset_index()
#     print(data_.head())
#     print(data_.shape)
    
    list_user_id = []
    list_item_similar = []
    list_score_similar = []
    all_list, list_user_id, list_item_similar = [], [], []
    for i, row in tqdm(data_.iterrows()):
        list_item_id = row['item_id'].split(',')    # 获取每一个用户的购买的item
        dict_item_id_score = {}
        
        self_items_list = []
        for i, item_i in enumerate(list_item_id[::-1]):
            for item_j, item_j_score in list(map(lambda x: [x[0], x[1]], deepwalk_model.wv.most_similar(positive=[item_i], topn=k))):
                
                if item_j not in dict_item_id_score:
                    dict_item_id_score[item_j] = 0
            
                beta = 0.8
                dict_item_id_score[item_j] += item_j_score / (1 + beta * i)

            
        dict_item_id_score_topk = sorted(dict_item_id_score.items(), key=lambda kv:kv[1], reverse=True)[:k]
        dict_item_id_set = set([item_similar for item_similar, score_similar in dict_item_id_score_topk])
            
            
        
        assert len(dict_item_id_set) == k, '召回的商品不为{k}个'.format(k=k)
        
        for item_similar, score_similar in dict_item_id_score_topk:
            list_item_similar.append(item_similar)
            list_score_similar.append(score_similar)
            list_user_id.append(row['user_id'])
        
    topk_recall = pd.DataFrame({'user_id':list_user_id,'item_similar':list_item_similar,'score_similar':list_score_similar})
    topk_recall['next_item_id'] = topk_recall['user_id'].map(dict_label)
    topk_recall['pred'] = topk_recall['user_id'].map(lambda x: 'train' if x in dict_label else 'test')
    
    return topk_recall


# In[7]:


# node2vec召回
def topk_recall_node2vec(
                            click_all, 
                            dict_label, 
                            node2vec_model, 
                            k=500
                            ):
    data_ = click_all.groupby(['user_id'])['item_id'].agg(lambda x:','.join(list(x))).reset_index()
#     print(data_.head())
#     print(data_.shape)
    
    list_user_id = []
    list_item_similar = []
    list_score_similar = []
    all_list, list_user_id, list_item_similar = [], [], []
    for i, row in tqdm(data_.iterrows()):
        list_item_id = row['item_id'].split(',')    # 获取每一个用户的购买的item
        dict_item_id_score = {}
        
        self_items_list = []
        for i, item_i in enumerate(list_item_id[::-1]):
            for item_j, item_j_score in list(map(lambda x: [x[0], x[1]], node2vec_model.wv.most_similar(positive=[item_i], topn=k))):
                
                if item_j not in dict_item_id_score:
                    dict_item_id_score[item_j] = 0
            
                beta = 0.8
                dict_item_id_score[item_j] += item_j_score / (1 + beta * i)

            
        dict_item_id_score_topk = sorted(dict_item_id_score.items(), key=lambda kv:kv[1], reverse=True)[:k]
        dict_item_id_set = set([item_similar for item_similar, score_similar in dict_item_id_score_topk])
            
            
        
        assert len(dict_item_id_set) == k, '召回的商品不为{k}个'.format(k=k)
        
        for item_similar, score_similar in dict_item_id_score_topk:
            list_item_similar.append(item_similar)
            list_score_similar.append(score_similar)
            list_user_id.append(row['user_id'])
        
    topk_recall = pd.DataFrame({'user_id':list_user_id,'item_similar':list_item_similar,'score_similar':list_score_similar})
    topk_recall['next_item_id'] = topk_recall['user_id'].map(dict_label)
    topk_recall['pred'] = topk_recall['user_id'].map(lambda x: 'train' if x in dict_label else 'test')
    
    return topk_recall


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




