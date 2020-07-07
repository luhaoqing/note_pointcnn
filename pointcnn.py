from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import pointfly as pf
import tensorflow as tf


def xconv(pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training, with_X_transformation, depth_multiplier,
          sorting_method=None, with_global=False):
    _, indices_dilated = pf.knn_indices_general(qrs, pts, K * D, True) # 带孔K*D近邻； (N, P, 3)  +  (N, P, 3)  => (N, P, K*D, 2) N=batch_num; P=point_num; 最后一维两个数：batch_indice + point_indice
    indices = indices_dilated[:, :, ::D, :] # (N, P, K, 2) 隔D取样，增加receptive field

    if sorting_method is not None:
        indices = pf.sort_points(pts, indices, sorting_method)

    nn_pts = tf.gather_nd(pts, indices, name=tag + 'nn_pts')  # (N, P, 3) + (N, P, K, 2) => (N, P, K, 3) indices维度比pts高，最后一维做索引，batch_indice->N, point_indice->P, 最后剩下(N, P, 3)里的3是坐标，补进去返回形状是 (N, P, K, 3)
    nn_pts_center = tf.expand_dims(qrs, axis=2, name=tag + 'nn_pts_center')  # (N, P, 1, 3)
    nn_pts_local = tf.subtract(nn_pts, nn_pts_center, name=tag + 'nn_pts_local')  # (N, P, K, 3)，减去中心点的坐标系，得到局部坐标系

    # Prepare features to be transformed
    # 2个全连接层 对点的相对坐标系升维，(N, P, K, 3) => (N, P, K, C_pts_fts)
    # 升维的目的是为了和前面的特征拼接 (ct.)
    nn_fts_from_pts_0 = pf.dense(nn_pts_local, C_pts_fts, tag + 'nn_fts_from_pts_0', is_training)
    nn_fts_from_pts = pf.dense(nn_fts_from_pts_0, C_pts_fts, tag + 'nn_fts_from_pts', is_training)
    # 如果第一次没有feature，就只使用从局部得到的点特征，如果有之前的特征，就拼接二者(ct.)
    if fts is None:
        nn_fts_input = nn_fts_from_pts
    else:
        nn_fts_from_prev = tf.gather_nd(fts, indices, name=tag + 'nn_fts_from_prev')
        nn_fts_input = tf.concat([nn_fts_from_pts, nn_fts_from_prev], axis=-1, name=tag + 'nn_fts_input') # 拼接坐标特征和之前的特征

    if with_X_transformation:
        ######################## X-transformation #########################
        # (N, P, K, 3)  > (N, P, 1, K * K)  ； kernel size 是(1, K)
        X_0 = pf.conv2d(nn_pts_local, K * K, tag + 'X_0', is_training, (1, K))
        # (N, P, 1, K * K) > (N, P, K, K)
        X_0_KK = tf.reshape(X_0, (N, P, K, K), name=tag + 'X_0_KK')
        # (N, P, K, K) > (N, P, 1, K * K)  可能为了较少参数量用depthwise_conv2d换conv2d
        X_1 = pf.depthwise_conv2d(X_0_KK, K, tag + 'X_1', is_training, (1, K))
        # (N, P, 1, K * K) > (N, P, K, K)
        X_1_KK = tf.reshape(X_1, (N, P, K, K), name=tag + 'X_1_KK')
        # (N, P, K, K) > (N, P, 1, K * K)
        X_2 = pf.depthwise_conv2d(X_1_KK, K, tag + 'X_2', is_training, (1, K), activation=None)
        # (N, P, 1, K * K) > (N, P, K, K)  最后的x矩阵 （K*K）
        X_2_KK = tf.reshape(X_2, (N, P, K, K), name=tag + 'X_2_KK')
        # 用得到的 x 矩阵乘以之前的特征，所以这里其实是有2个分支，一个计算x矩阵,另外一个计算特征；(ct.)
        # (N, P, K, K) * (N, P, K, C_pts_fts) = (N, P, K, C_pts_fts)
        fts_X = tf.matmul(X_2_KK, nn_fts_input, name=tag + 'fts_X')
        ###################################################################
    else:
        fts_X = nn_fts_input

    # 最后的分离卷积
    fts_conv = pf.separable_conv2d(fts_X, C, tag + 'fts_conv', is_training, (1, K), depth_multiplier=depth_multiplier) # (N, P, K, C_pts_fts) -> (N, P, 1, C)
    fts_conv_3d = tf.squeeze(fts_conv, axis=2, name=tag + 'fts_conv_3d') # (N, P, C)

    # 用代表点全局位置信息,
    if with_global:
        fts_global_0 = pf.dense(qrs, C // 4, tag + 'fts_global_0', is_training)
        fts_global = pf.dense(fts_global_0, C // 4, tag + 'fts_global', is_training) # (N, P, C//4)  最后一层时作者使receptive field < 1让model更会捕捉local feature; 从“subvolume supervision”得到启发 to further address the over-fitting problem.;
        return tf.concat([fts_global, fts_conv_3d], axis=-1, name=tag + 'fts_conv_3d_with_global')
    else:
        return fts_conv_3d


class PointCNN:
    def __init__(self, points, features, is_training, setting):
        xconv_params = setting.xconv_params
        fc_params = setting.fc_params
        with_X_transformation = setting.with_X_transformation
        sorting_method = setting.sorting_method
        N = tf.shape(points)[0]

        if setting.sampling == 'fps':
            from sampling import tf_sampling

        self.layer_pts = [points]
        if features is None:
            self.layer_fts = [features]
        else:
            features = tf.reshape(features, (N, -1, setting.data_dim - 3), name='features_reshape')
            C_fts = xconv_params[0]['C'] // 2
            features_hd = pf.dense(features, C_fts, 'features_hd', is_training)
            self.layer_fts = [features_hd]

        for layer_idx, layer_param in enumerate(xconv_params):
            tag = 'xconv_' + str(layer_idx + 1) + '_'
            K = layer_param['K']
            D = layer_param['D']
            P = layer_param['P']
            C = layer_param['C']
            links = layer_param['links']
            if setting.sampling != 'random' and links:
                print('Error: flexible links are supported only when random sampling is used!')
                exit()

            # get k-nearest points
            pts = self.layer_pts[-1]
            fts = self.layer_fts[-1]
            # -----------sampling, 选择代表点qrs-----------------
            if P == -1 or (layer_idx > 0 and P == xconv_params[layer_idx - 1]['P']): # all points入选, 如果第一层且p与前一层数相同；or setting的这一层P=-1
                qrs = self.layer_pts[-1]
            else:
                if setting.sampling == 'fps': # segmentation
                    fps_indices = tf_sampling.farthest_point_sample(P, pts)
                    batch_indices = tf.tile(tf.reshape(tf.range(N), (-1, 1, 1)), (1, P, 1))
                    indices = tf.concat([batch_indices, tf.expand_dims(fps_indices,-1)], axis=-1)
                    qrs = tf.gather_nd(pts, indices, name= tag + 'qrs') # (N, P, 3)
                elif setting.sampling == 'ids':
                    indices = pf.inverse_density_sampling(pts, K, P)
                    qrs = tf.gather_nd(pts, indices)
                elif setting.sampling == 'random': # classification
                    qrs = tf.slice(pts, (0, 0, 0), (-1, P, -1), name=tag + 'qrs')  # (N, P, 3)
                else:
                    print('Unknown sampling method!')
                    exit()
            self.layer_pts.append(qrs)

            # ------------------选定channel数和depth_multiplier， ---------------------------
            if layer_idx == 0:
                C_pts_fts = C // 2 if fts is None else C // 4 # how选择channel数
                depth_multiplier = 4
            else:
                C_prev = xconv_params[layer_idx - 1]['C']
                C_pts_fts = C_prev // 4
                depth_multiplier = math.ceil(C / C_prev)
            with_global = (setting.with_global and layer_idx == len(xconv_params) - 1) # segmentation会在xconv的最后一层加上全局位置
            fts_xconv = xconv(pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training, with_X_transformation,
                              depth_multiplier, sorting_method, with_global)

            # ------------------list, 用之前层的feature---------------------------
            fts_list = []
            for link in links:
                fts_from_link = self.layer_fts[link]
                if fts_from_link is not None:
                    fts_slice = tf.slice(fts_from_link, (0, 0, 0), (-1, P, -1), name=tag + 'fts_slice_' + str(-link))
                    fts_list.append(fts_slice)
            if fts_list:
                fts_list.append(fts_xconv)
                self.layer_fts.append(tf.concat(fts_list, axis=-1, name=tag + 'fts_list_concat'))
            else:
                self.layer_fts.append(fts_xconv)

        # ------------------xdconv 层用在segmentation---------------------------
        if hasattr(setting, 'xdconv_params'):
            for layer_idx, layer_param in enumerate(setting.xdconv_params):
                tag = 'xdconv_' + str(layer_idx + 1) + '_'
                K = layer_param['K']
                D = layer_param['D']
                pts_layer_idx = layer_param['pts_layer_idx']
                qrs_layer_idx = layer_param['qrs_layer_idx']

                pts = self.layer_pts[pts_layer_idx + 1]
                fts = self.layer_fts[pts_layer_idx + 1] if layer_idx == 0 else self.layer_fts[-1]
                qrs = self.layer_pts[qrs_layer_idx + 1]
                fts_qrs = self.layer_fts[qrs_layer_idx + 1]
                P = xconv_params[qrs_layer_idx]['P']
                C = xconv_params[qrs_layer_idx]['C']
                C_prev = xconv_params[pts_layer_idx]['C']
                C_pts_fts = C_prev // 4
                depth_multiplier = 1
                fts_xdconv = xconv(pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training, with_X_transformation,
                                   depth_multiplier, sorting_method)
                fts_concat = tf.concat([fts_xdconv, fts_qrs], axis=-1, name=tag + 'fts_concat')
                fts_fuse = pf.dense(fts_concat, C, tag + 'fts_fuse', is_training)
                self.layer_pts.append(qrs)
                self.layer_fts.append(fts_fuse)
        # ------------------dropout---------------------------
        self.fc_layers = [self.layer_fts[-1]]
        for layer_idx, layer_param in enumerate(fc_params):
            C = layer_param['C']
            dropout_rate = layer_param['dropout_rate']
            fc = pf.dense(self.fc_layers[-1], C, 'fc{:d}'.format(layer_idx), is_training)
            fc_drop = tf.layers.dropout(fc, dropout_rate, training=is_training, name='fc{:d}_drop'.format(layer_idx))
            self.fc_layers.append(fc_drop)
