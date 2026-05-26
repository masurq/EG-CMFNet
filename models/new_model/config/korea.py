from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from easydict import EasyDict as edict
import argparse

C = edict()
config = C

# 设置训练日志
C.log_path = r''
C.env = 'train'
C.log_append = False
C.reduced_log_append = False

# 训练设置
C.train_batch_size =16#16
C.in_channel1 = 3
C.in_channel2 = 1
C.use_amp = False
C.backbone ='biformer-t'
C.pretrained_model = 'pre_trained/biformer/biformer_tiny_best.pth'
C.decoder ='MLPAlignDecoder'
C.decoder_embed_dim = 256

C.start_epoch = 0
C.total_epoch = 100

C.loss_names = 'cross_entropy'
C.aux_rate = 0.4
C.num_classes = 6
C.label_smoothing = 0
C.class_loss_weight = [1, 1, 1, 1, 1, 1]
C.loss_weight_with_background = True
C.loss_ignore_index =6
C.val_with_background =1

C.best_acc_metric = 'Pixel_Acc'

# val设置
C.val_batch_size = 2 # 36
C.eval = False
C.no_val = False
C.show_val_image = False

"""Eval Config"""
C.test_batch_size =128#128

C.norm_mean = [0.24754184923558617,
0.22875652002781224,
0.21594419016553776],
C.norm_std =   [0.1091438337814174,
0.11703869373910572,
0.1209566808804282 ]
C.norm_mean_X = [0.19897191111648427]
C.norm_std_X = [0.17366085451003405]

C.eval_stride_rate = 1 / 2
C.eval_crop_size = [256, 256]  # [height weight]

C.eval_scale_array = [1]#[0.75, 1, 1.25]  # [0.5, 0.75, 1.0, 1.25, 1.5] [0.75, 1.0, 1.25, 1.5, 1.75] [0.75, 1.0, 1.25, 1.5]
C.eval_flip = True  # False

C.model_path = r''
C.pred_path =  ''

C.resume_model = False
C.resume_model_path = ''
C.resume_start_epoch = 0
C.resume_total_epoch = 500

# dataset设置
C.dataset_name = 'sixx'
C.train_data_root = r'run_data/korea/korea/train'
C.val_data_root = r'run_data/korea/korea/val'
C.test_data_root = r'run_data/korea/korea/test'

C.aug_flip = True  # 是否使用随机翻转
C.aug_rotate = True  # 是否使用随机旋转
C.aug_crop = False  # 是否使用随机裁剪
C.aug_crop_size = 256  # 随机裁剪大小
C.aug_hsv = False  # 是否使用随机HSV变换
C.aug_hue_limit = (-180, 180)  # 随机Hue变换范围
C.aug_sat_limit = (-255, 255)  # 随机Saturation变换范围
C.aug_val_limit = (-255, 255)  # 随机Value变换范围
C.aug_shift_scale_rotate = True  # 是否使用随机平移缩放旋转
# scale和aspect都加了1
C.aug_ssr_shift_limit = (-0.5, 0.5)  # 随机平移变换范围
C.aug_ssr_scale_limit = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75)  # 随机缩放变换范围
C.aug_ssr_rotate_limit = (-0.5, 0.5)  # 随机旋转变换范围
C.aug_ssr_aspect_limit = (-0.5, 0.5)  # 随机拉伸变换范围
C.aug_brightness = False  # 是否使用随机亮度变换
C.aug_bright_limit = (-255, 255)  # 随机亮度变换范围
C.aug_noise = False  # 是否使用随机噪声
C.aug_noise_mode = 'gaussian'  # 随机噪声类型
C.aug_blur = False  # 是否使用随机滤波
C.aug_blur_mode = 'Median'  # 随机滤波模式
C.aug_blur_limit = 7  # 随机滤波卷积核大小r
C.cutout = False  # 不适合vaihingen和potsdam数据集
C.n_holes = 1
C.cut_size = (16, 16)

# 环境设置
C.use_cuda = True
C.gpu_ids = [0]
C.num_workers = 10
C.random_seed = 42  # 3407/114514
C.save_root = r''
C.experiment_start_time = time.strftime('%m-%d-%H-%M-%S', time.localtime(time.time()))

# 优化器和学习率设置/media/ubuntu/7c2c9e84-2646-43cd-8669-ef651d32fd3b/zm/ZMSeg/checkpoints/korea/fanxiu/fcnnopara/MLPAlignDecoder_biformer-t/all_train/12-10-09-31-07/MLPAlignDecoder_biformer-t_all_train_epoch102.pth
C.lr = 1e-4
C.lr_power = 0.9
C.use_lr_warmup = True  # 是否启用学习率WarmUp策略
C.lr_warmup_init = 5e-5  # WarmUp策略初始学习率5e-5
C.lr_warmup_epoch = 10

C.lr_scheduler ='CosineAnnealingLR'#'ReduceLROnPlateau'

# ReduceLROnPlateau,自适应调整学习率，有min和max两种模式
# min 表示当指标不再降低(如监测loss)， max 表示当指标不再升高(如监测 accuracy)。
C.lr_reduce_metrics = 'Pixel_Acc'  # 自适应评价指标,Pixel_Acc,mIoU, Val_Loss, 匹配ReduceLROnPlateau
C.lr_reduce_mode = 'max'  # 自适应评价模式
C.lr_reduce_factor = 0.1  # 自适应更新系数，调整倍数，类似gamma，更新率为lr=lr*factor
C.lr_reduce_patience = 8  # 自适应更新耐心，注意这里的max初始指标是从warm_up之后的epoch开始计算的，不包括warm_up中的结果
C.lr_reduce_threshold = 1e-4  # 自适应容差阈值
C.lr_reduce_cooldown = 0  # 自适应冷却时间conv357911
C.lr_reduce_min_lr = 1e-6  # 自适应最小学习率

C.lr_gamma = 0.1

# 'CosineAnnealingLR'
C.lr_T_max = 20  # 半周期
C.lr_eta_min = 0  # 5e-6

# 'CosineAnnealingWarmRestarts',相当于没有缓慢上升的过程，而是直接在restart_epoch回到最大值
C.lr_T_0 =15 #15 # 学习率第一次回到初始值的epoch位置
C.lr_T_mult = 2  # 控制学习率回升的速度，r如果mult=2，则学习率在5,15,35,75...处回到最大值,前一个值*2+5

C.lr_milestones = [90, 130]  # 'MultiStepLR'

C.lr_step_size = 20  # 'StepLR'

C.optimizer_name = 'Adam'
C.momentum = 0.5  # 动量,用于SGD,0.9
C.weight_decay = 1e-4  # 5e-4,用于SGD

C.bn_eps = 1e-5
C.bn_momentum = 0.1


def open_tensorboard():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()

    if args.tensorboard:
        open_tensorboard()
