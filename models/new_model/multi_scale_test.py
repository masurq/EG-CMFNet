import os
import shutil
import time
from tqdm import tqdm
import numpy as np
from prettytable import PrettyTable
import torch
import torch.nn as nn
from utils import metrics
from utils.Common import common
from utils.Logger import Logger
import utils.Info as Info
from utils import colorize_mask, Evaluator

from datasets.Opt_SarDataset import Opt_SarDataset as Dataset
from config.opt_sar_config512 import config
from builder import EncoderDecoder as segmodel

if config.dataset_name == 'eight':
    from utils import eight_classes

    class_name = eight_classes()
elif config.dataset_name == 'six':
    from utils import six_classes

    class_name = six_classes()
elif config.dataset_name == 'seven':
    from utils import seven_classes

    class_name = seven_classes()


def snapshot_forward(logger, dataset, num_classes, segmentor, save_path=None):
    conf_mat = np.zeros((num_classes, num_classes)).astype(np.int64)
    tbar = tqdm(dataset)
    for index, data in enumerate(tbar):
        with torch.no_grad():
            imgs = data[0]
            imgs_X = data[1]
            masks = data[2]
            output_names = data[3]

            preds = segmentor.sliding_eval_rgbX(imgs, imgs_X, config.eval_crop_size,
                                                config.eval_stride_rate, 0)


            if save_path is not None:
                pred_vis_pil = colorize_mask(preds, dataset_name=config.dataset_name)
                pred_save_path = os.path.join(save_path, 'predict')

                if not os.path.exists(pred_save_path):
                    os.makedirs(pred_save_path)
                pred_vis_pil.save(os.path.join(pred_save_path, output_names.replace('.tif', '.png')))

            conf_mat += metrics.confusion_matrix(pred=preds,
                                                 label=masks,
                                                 num_classes=num_classes)

    test_acc, test_kappa, test_mean_IoU, test_mean_f1, test_mean_precision, test_mean_recall, \
    test_IoU, test_f1_score, test_precision, test_recall = metrics.evaluate(conf_mat, 1, config.val_with_background)

    logger.log('None', ' Pixel Acc:      {}'.format(str(test_acc)), show_time=False)
    logger.log('None', ' Kappa:          {}'.format(str(test_kappa)), show_time=False)
    logger.log('None', ' Mean IoU:       {}'.format(str(test_mean_IoU)), show_time=False)
    logger.log('None', ' Mean F1 score:  {}'.format(str(test_mean_f1)), show_time=False)
    logger.log('None', ' Mean Precision: {}'.format(str(test_mean_precision)), show_time=False)
    logger.log('None', ' Mean Recall:    {}'.format(str(test_mean_recall)), show_time=False)

    table = PrettyTable(["序号", "名称", "IoU", "F1 Score", "Precision", "Recall"])
    _class_name = class_name
    if config.val_with_background == 0:
        _class_name = _class_name[1:]
    for i in range(len(_class_name)):
        table.add_row(
            [i, _class_name[i], '{:.8f}'.format(test_IoU[i]), '{:.8f}'.format(test_f1_score[i]),
             '{:.8f}'.format(test_precision[i]), '{:.8f}'.format(test_recall[i])])

    logger.log('None', '{}'.format(table), show_time=False)


def reference():
    log_env = '{}/multi_test_results/{}_{}_{}'.format(config.log_path, config.decoder, config.backbone, config.env)
    common.check_path(log_env)
    log_filename = '{}/{}_{}_{}_multi_test.log'.format(log_env, config.decoder, config.backbone, config.env)
    logger = Logger(log_filename, append=config.log_append)

    Info.info_logger(logger)
    logger.log('ZMSegmentation', '{}_{}_{}'.format(config.decoder, config.backbone, config.env), show_time=True,
               print_type='print')

    if config.use_cuda:
        logger.log('INFO', 'Numbers of GPUs:{}'.format(len(config.gpu_ids)), show_time=False)
    else:
        logger.log('INFO', 'Using CPU', show_time=False)

    dataset = Dataset(class_name=class_name, root=config.test_data_root, img_opt_transform=None,
                      img_aux_transform=None, mask_transform=None, mode='test')

    logger.log('INFO', '类别数：{}'.format(len(class_name)), show_time=False)
    logger.log('INFO', '{}'.format(class_name), show_time=False)

    model = segmodel(cfg=config, logger=logger, norm_layer=nn.BatchNorm2d)

    logger.log('INFO', '=====> model {}_{} =============== '.format(config.decoder, config.backbone),
               show_time=True)

    state_dict = torch.load(config.model_path)
    logger.log('INFO', 'Loading Model from {}'.format(config.model_path), show_time=False, print_type='print')
    model.load_state_dict(state_dict)

    logger.log('INFO', '===============> load model successfully', show_time=True, print_type='print')

    if not config.pred_path:
        pass
    elif not os.path.exists(config.pred_path):
        os.makedirs(config.pred_path)
    else:
        shutil.rmtree(config.pred_path)
        os.makedirs(config.pred_path)

    segmentor = Evaluator(len(class_name), config.norm_mean, config.norm_std, config.norm_mean_X, config.norm_std_X,
                          model, config.eval_scale_array, config.eval_flip)
    t_start = time.time()

    snapshot_forward(logger, dataset, len(class_name), segmentor, config.pred_path)

    t_end = time.time()
    elapsed_time = t_end - t_start
    hours = int(elapsed_time / 3600)
    minutes = int((elapsed_time % 3600) / 60)
    seconds = int((elapsed_time % 3600) % 60)
    logger.log('INFO', 'Multi-scale test task finished.', show_time=True, print_type='print')
    logger.log('INFO', 'Total time consuming: {}时{}分{}秒'.format(hours, minutes, seconds),
               show_time=True)


if __name__ == '__main__':
    reference()
