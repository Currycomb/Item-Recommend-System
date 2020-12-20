#!/usr/bin/env python
# coding: utf-8

# ## node2vec和deepwalk召回

# In[26]:


import os
import time
import random
import itertools
import numpy as np
import pandas as pd 
import networkx as nx
from gensim.models import Word2Vec, KeyedVectors
from joblib import Parallel, delayed


# In[20]:


random.seed(2020)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option("display.max_colwidth",100)
pd.set_option('display.width',1000)
now_phase = 0
user_data = '../../data/'


# In[3]:


def create_alias_table(area_ratio):
    """
    :param area_ratio: sum(area_ratio)=1
    :return: accept,alias
    """
    l = len(area_ratio)
    accept, alias = [0] * l, [None] * l
    small, large = [], []
    area_ratio_ = np.array(area_ratio) * l
    for i, prob in enumerate(area_ratio_):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:

        small_idx, large_idx = small.pop(), large.pop()

        accept[small_idx] = area_ratio_[small_idx]

        alias[small_idx] = large_idx
        area_ratio_[large_idx] = area_ratio_[large_idx] - (1 - area_ratio_[small_idx])

        if area_ratio_[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    while large:
        large_idx = large.pop()
        accept[large_idx] = 1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1
    return accept, alias


# In[4]:


def alias_sample(accept, alias):
    """
    :param accept:
    :param alias:
    :return: sample index
    """
    N = len(accept)
    i = int(np.random.random() * N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]


# In[5]:


def partition_num(num, workers):
    if num % workers == 0: 
        return [num // workers] * workers
    else: 
        return [num // workers] * workers + [num % workers]


# In[6]:


class RandomWalker:

    def __init__(self, G, p=1, q=1):
        """
        :param G:
        :param p: Return parameter,controls the likelihood of immediately revisiting a node in the walk.
        :param q: In-out parameter,allows the search to differentiate between “inward” and “outward” nodes
        """
        self.G = G
        self.p = p
        self.q = q

    def deepwalk_walk(self, walk_length, start_node):
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    def node2vec_walk(self, walk_length, start_node):
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                #由于node2vec采样需要cur节点v，prev节点t，所以当没有前序节点时，直接使用当前顶点和邻居顶点之间的边权作为采样依据
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    edge = (prev, cur)
                    next_node = cur_nbrs[alias_sample(alias_edges[edge][0],alias_edges[edge][1])]
                    walk.append(next_node)
            else: 
                break
        return walk

    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):
        """
        """
        G = self.G
        nodes = list(G.nodes())
        print(partition_num(num_walks, workers))
        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length) for num in
            partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))
        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length, ):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                if self.p == 1 and self.q == 1:
                    walks.append(self.deepwalk_walk(
                        walk_length=walk_length, start_node=v))
                else:
                    walks.append(self.node2vec_walk(
                        walk_length=walk_length, start_node=v))
        return walks

    def get_alias_edge(self, t, v):
        """
        compute unnormalized transition probability between nodes v and its neighbors give the previous visited node t.
        :param t:
        :param v:
        :return:
        """
        G = self.G
        p = self.p
        q = self.q
        unnormalized_probs = []
        for x in G.neighbors(v):
            weight = G[v][x].get('weight', 1.0)  # w_vx
            if x == t:  # d_tx == 0
                unnormalized_probs.append(weight/p)
            elif G.has_edge(x, t):  # d_tx == 1
                unnormalized_probs.append(weight)
            else:  # d_tx > 1
                unnormalized_probs.append(weight/q)

        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
        return create_alias_table(normalized_probs)

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        G = self.G
        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr].get('weight', 1.0) for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = create_alias_table(normalized_probs)
        alias_edges = {}
        for edge in G.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges


# In[7]:


class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, workers=1):

        self.graph = graph
        self.w2v_model = None
        self._embeddings = {}

        self.walker = RandomWalker(graph, p=1, q=1, )
        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    def train(self, embed_size=128, window_size=5, workers=3, iters=5, **kwargs):

        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iters

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model
        return model

    def get_embeddings(self, ):
        if self.w2v_model is None:
            print("model not train")
            return {}
        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]
        return self._embeddings

    def get_topK(self, item, k=50):
        if not isinstance(item, str):
            item=str(item)
        recom_list = list(map(lambda x: [x[0], x[1]], self.w2v_model.wv.most_similar(positive=[item], topn=k)))
        return recom_list


# In[8]:


class Node2Vec:

    def __init__(self, graph, walk_length, num_walks, p=1.0, q=1.0, workers=1):

        self.graph = graph
        self._embeddings = {}
        
        self.walker = RandomWalker(graph, p=p, q=q, )
        self.walker.preprocess_transition_probs()
        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    def train(self, embed_size=128, window_size=5, workers=3, iters=5, **kwargs):

        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1
        kwargs["hs"] = 0  # node2vec not use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iters

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")
        self.w2v_model = model

        return model

    def get_embeddings(self,):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings

    def get_topK(self, item, k=50):
        if not isinstance(item, str):
            item = str(item)
        recom_list = list(map(lambda x: [x[0], x[1]], self.w2v_model.wv.most_similar(positive=[item], topn=k)))
        return recom_list


# In[9]:


def get_item_graph(df, user_col, item_col, direction=True, new_wei=False):
    """构造图
    """
    user_item_ = df.groupby(user_col)[item_col].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))
    edgelist = []
    user_time_ = df.groupby(user_col)['time'].agg(list).reset_index() # 引入时间因素
    user_time_dict = dict(zip(user_time_[user_col], user_time_['time']))

    item_cnt=df[item_col].value_counts().to_dict()

    for user, items in user_item_dict.items():
        for i in range(len(items) - 1):
            if direction:
                t1 = user_time_dict[user][i] # 点击时间提取
                t2 = user_time_dict[user][i+1]
                delta_t=abs(t1-t2)*50000   # 中值 0.01 75%:0.02
                #             有向有权图，热门商品-->冷门商品权重=热门商品个数/冷门商品个数
                ai, aj = item_cnt[items[i]], item_cnt[items[i+1]]
                edgelist.append([items[i], items[i + 1], max(3, np.log(1+ai/aj)) * 1/(1+delta_t) ])
                edgelist.append([items[i+1], items[i], max(3, np.log(1+aj/ai)) * 0.8 * 1/(1+delta_t) ])
            else:
                edgelist.append([items[i], items[i + 1], 1])
    if direction:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    for edge in edgelist:
        G.add_edge(str(edge[0]), str(edge[1]), weight=edge[2])
    if new_wei:
        for u,v,d in G.edges(data=True):
            deg = G.degree(u)/G.degree(v)
            if deg < 1:
                deg = max(0.1, deg)
            else:
                deg = min(3, deg)
            new_weight = d["weight"] * deg
            G[u][v].update({"weight":new_weight})
    return G


# In[22]:


def deep_node_recom():
    """使用全量数据分别训练deepwalk和node2vec模型
    """
    global now_phase
    novalid_click = pd.DataFrame()
    whole_click = pd.DataFrame()
    for i in range(now_phase+1):
        click_train=pd.read_csv(user_data+'underexpose_train/underexpose_train_click-{}.csv'.format(i),header=None, names=['user_id', 'item_id', 'time'])
        click_test=pd.read_csv(user_data+'underexpose_test/underexpose_test_click-{}.csv'.format(i),header=None, names=['user_id', 'item_id', 'time'])
        qtime_test=pd.read_csv(user_data+'underexpose_test/underexpose_test_qtime-{}.csv'.format(i),header=None, names=['user_id', 'time'])
        click_train["time"] += i
        click_test["time"] += i
        qtime_test["time"] += i
        all_click = click_train.append(click_test)
        novalid_click = novalid_click.append(all_click)    # 不包含要预测的数据
        all_click = all_click.append(qtime_test)
        whole_click = whole_click.append(all_click)


    """除去test最后一次点击的whole点击数据，用于offline的召回
    """
    novalid_click = novalid_click.drop_duplicates(subset=['user_id', 'item_id', 'time'], keep='last')
    novalid_click = novalid_click.sort_values('time')
    novalid_click = novalid_click.reset_index(drop=True)
    print('novalid_click')
    print(novalid_click.shape)
    
    """whole点击数据
    """
    whole_click = whole_click.drop_duplicates(subset=['user_id', 'item_id', 'time'], keep='last')
    whole_click = whole_click.sort_values('time')
    whole_click = whole_click.reset_index(drop=True)
    cpu_jobs = os.cpu_count() - 1
    
    """使用有向图训练的node2vec
    """
    G = get_item_graph(novalid_click, 'user_id', 'item_id')
    novalidmodel = Node2Vec(G, walk_length=20, num_walks=80, p=2, q=0.5, workers=1)
    novalidmodel.train(embed_size=128, window_size=10, workers=cpu_jobs, iter=3)
    novalidmodel.w2v_model.wv.save_word2vec_format(user_data + "node2vec_offline.bin", binary=True)

    """deepwalk
    """
    G = get_item_graph(novalid_click, 'user_id', 'item_id', direction=False)
    novalidmodel = DeepWalk(G, walk_length=20, num_walks=80, workers=8)
    novalidmodel.train(embed_size=128, window_size=10, workers=cpu_jobs, iter=3)
    novalidmodel.w2v_model.wv.save_word2vec_format(user_data + "deepwalk_offline.bin", binary=True)


# In[23]:


if __name__ == "__main__":
    print("start")
    a = time.time()
    if not os.path.exists(user_data):
        os.mkdir(user_data)

    deep_node_recom()
    print("time:{:6.4f} mins".format( (time.time()-a)/60))


# ## 加载模型

# In[27]:


deepwalk_model = KeyedVectors.load_word2vec_format(user_data + 'deepwalk_offline.bin', binary=True)
node2vec_model = KeyedVectors.load_word2vec_format(user_data + 'node2vec_offline.bin', binary=True)


# In[35]:


recom_list = list(map(lambda x: [x[0], x[1]], deepwalk_model.wv.most_similar(positive=[item], topn=50)))
recom_list


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




