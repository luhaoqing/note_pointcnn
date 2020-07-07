# note_pointcnn

## pointcnn's hierarchical convolution
Pointcnn在点集pts中取样得到代表点qrs, 每个代表点找到k近邻，从k近邻点和该代表点的相对位置得到坐标特征，和代表点本身的特征fts拼一起，得到该代表点的总特征   
用总特征做输入训练x-matrix，x-matrix负责weight and permute总特征来解决unordered的问题   
经过x-matrix和最后一层separableconv(让dim对的上)后的feature作为这一轮代表点的本身的特征，成为下一轮的点集  
(抓一群点，从k近邻提炼特征，下一轮从这群点里再抓新的一群点，上一轮的特征这一轮会参考，还可以用list联系参考好几轮之前的特征，点越来越少，特征越来越明显)  
#### process
1. 取样: random if classification, furthest point if segmentation
2. 选定channel数和depth_multiplier
3. [x-conv](#x-conv)获得features
4. 有list就combine features in previous layers
5. X-dconv 可能用在segmentation里
6. Droppout  


## x-conv
```python3
def xconv(pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training, with_X_transformation, depth_multiplier,
          sorting_method=None, with_global=False):
```
  Pts=points set  
  Fts=features set corresponding to points set  
  Qrs=representative points for this point cloud figure  
  K=k nearest neighbour  
  D=dilation rate  
  C=最后output的channel数  
  C_pts_fts=coordinate feature的channel数  
  with_X_transformation=weighted and permuted by the X -transformation，比较，作者得到X-transform提高了准确度  
  Depth_multiplier=para in depthwise conv and separable conv  
  Sorting method=k nearest neighbour  
  With_global=whether append global pos info after last x conv layer if segmentation  

#### process
1. 获得代表点(qrs)的k近邻  
    * how: K*D nearest neighbour; 带孔D   
    * why: 增加receptive field  
2. Knn’s relative positions w.r.t representative point(qrs) 
    * why: ‘X -Conv is designed to work on local point regions, and the output should not be dependent on the absolute position of p and its neighboring points, but on their relative positions.’
3. Lifting relative coordinates (step2) into coordinate features(Dim=(N, P, K, C_pts_fts)) 
    * how: 2个全连接层  
    * why: 升维后可以和fts拼接  
4. 拼接坐标特征和之前的特征(总特征nn_fts_input)，如果是第一层，既没有extra_features，又没有之前层获得的feature，在这层就只用坐标特征
5. **X-transformation** 训练X matrix   
    * how: 一层conv2d, 两层separable-conv2d  
6. X-transformation*总特征(step 4) = fts_X
7. 最后一层维度从(N, P, K, C_pts_fts) 到 (N, P, C)得到输出特征矩阵  
    * how: separable-conv2d + squeeze
8. 如果segmentation, 可以在最后一层xconv后加代表点全局位置信息 
    * why: 'Harvest the global position information of the representative points in the last X -Conv layer'; receptive field supposed to be less than 1
  
## overfitting
1. dropout  
2. with_global  
3. depthwise convolution, 也加快了计算  

## data augmentation
1. 'To train a model that takes N points as input, N (N,(N/8)2) points are used for training, where N denotes a Gaussian distribution.'
2. 'randomly sample and shuffle the input points, such that both the neighboring point sets and order may differ from batch to batch.'

## potential improvement
1. 作者觉得采样方法可以改进    
2. 作者写到 ’X -transformations are far from ideal, especially in terms of the permutation’  

## reference
https://blog.csdn.net/sinat_37532065/article/details/83001494  
https://zhuanlan.zhihu.com/p/76915660

