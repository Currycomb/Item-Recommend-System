# 商品top50推荐系统
## 问题建模
本项目的数据集给出了15万左右的用户以及12万左右的商品, 以及对应的经过脱敏处理的用户特征和经过预处理的商品特征，旨在为用户推荐50个其可能购买的商品。
## 推荐系统架构方案
  本项目采用传统的召回＋排序的方案。在召回模块采用deepwalk， node2vec，item_feature, itemCF四种方法进行多路召回，为每位用户召回1000个商品。在排序阶段采用wide&deep模型，对召回的1000个商品进行排序。将排序所得的分数依据商品点击量进行后处理，来增大对非热门商品的曝光度。最后根据处理后的分数为每位用户推荐50个商品。

**最终实现了在验证集上top50召回率0.807， 测试集上top50召回率0.712**


## 文件结构
数据来源于阿里天池平台开源数据，在百度网盘里面，可以自行下载，按照以下路径创建文件夹以及放置数据。

百度网盘链接：https://pan.baidu.com/s/1sspNWKYVxf-QFTrCjdqfoQ 
提取码：853t 

    │  feature_list.csv                               # List the features we used in ranking process
    │  project_structure.txt                          # The tree structure of this project
    ├─ build_graph_model.py                          # Build deepwalk model and node2vec model
    ├─ final_rank.py                          # Build wide&deep network
    ├─ final_solution.py                          # Main program
    ├─ recall_function.py                          # Functions used to recall items
    ├─ item_feat.pkl                          # Item feature after PCA
    ├─ top100_recall_feature.pkl                          # Recalled 100 items for each user by using item_feature
    ├─ top300_recall_deepwalk_result.pkl                          # Recalled 300 items for each user by using deepwalk
    ├─ top300_recall_node2vec_result.pkl                          # Recalled 300 items for each user by using node2vec
    ├─ topk_recall.pkl                          # Recalled 1000 items for each user by combining all ways
    ├─ train_eval_rank.pkl                          # Cross validation set after ranking
    ├─ wide_and_deep.h5                          # Wide&Deep model using full training set
    ├─ wide_and_deep_no_cv.h5                          # Wide&Deep model using training set except cross validation set
    ├─ data                                           # Origin dataset
    │  ├─ underexpose_test
    │  └─ underexpose_train
    ├─ readme.md
    ├─ deepwalk_offline.bin                                      # deepwalk model
    └─ node2vec_offline.bin                                      # node2vec model

## Python库环境依赖

    tensorflow==2.3.1
    scikit-learn==0.23.2
    joblib==0.17.0
    networkx==2.1
    gensim==3.8.3
    pandas==0.25.1
    numpy==1.18.5
    tqdm==4.26.0

## 声明
本项目所有代码仅供各位同学学习参考使用。


