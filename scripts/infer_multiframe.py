"""
Script for running inference for tool-tip/pose segmentation models 
Author: Bhargav Ghanekar
"""

import os 
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
sys.path.append('.')
sys.path.append('./models/')
import logging, json, random, time, math 
from pathlib import Path

import configargparse
from configs.config_multiframe import test_config_parser as config_parser

import cv2 
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms

import matplotlib.pyplot as plt
from src.dataloader_multiframe import get_data_loader
from src.metrics import get_metrics
from models import get_multiframe_segmentation_model as get_model
from utils.dataloader_utils import get_MICCAI2017_dataset_filenames, get_JIGSAWS_dataset_filenames, get_MICCAI2015_dataset_filenames
from utils.dataloader_utils import get_custom_dataset_filenames
from utils.log_utils import AverageMeter, ProgressMeter
from utils.model_utils import load_model_weights
from utils.train_utils import add_metrics_meters
from utils.vis_utils import mask_overlay, draw_plus
from utils.localization_utils_v2 import centroid_error

def main(): 
    parser = configargparse.ArgumentParser() 
    parser = config_parser(parser) 
    args = parser.parse_args() 
    main_worker(args) 

def test(dataloader, model, args, file_names, logger, writer=None, optflow_model=None): 
    if args.add_optflow_inputs: 
        assert optflow_model is not None, "Optical flow model should be provided"
        optflow_model.eval()
    logging.info(f'Testing/Infering on {args.dataset} dataset')
    batch_time = AverageMeter(' Forward Time', ':2.2f')
    data_time = AverageMeter(' Data Time', ':2.2f') 
    progress_meter_list = [batch_time, data_time] 
    progress_meter_list = add_metrics_meters(progress_meter_list, args.metric_fns, args.num_classes) 
    progress = ProgressMeter(len(dataloader), progress_meter_list, prefix='Test: ')
    model.eval()
    data_time_start = time.time()
    step = 0 
    centroid_pred_err_rt = []
    centroid_pred_err_rb = []
    centroid_pred_err_lt = []
    centroid_pred_err_lb = []
    centroid_pres_err_rt = []
    centroid_pres_err_rb = []
    centroid_pres_err_lt = []
    centroid_pres_err_lb = []
    with torch.no_grad():
        for sample in dataloader: 
            data_time.update(time.time() - data_time_start)
            batch_time_start = time.time() 
            if torch.cuda.is_available():
                input = [sample['input'][i].cuda(non_blocking=True) for i in range(len(sample['input']))]
                mask = sample['mask'].type(torch.LongTensor).cuda(non_blocking=True)
                if args.add_depth_inputs:
                    input_depth = [sample['input_depth'][i].cuda(non_blocking=True) for i in range(len(sample['input_depth']))]
            else: 
                mask = sample['mask'].type(torch.LongTensor)
            mask = mask.squeeze(1)
            if args.add_optflow_inputs:
                optflow = [] 
                frame0 = F.interpolate(input[0], scale_factor=1.0, mode='nearest')
                if args.optflow_model == 'FlowFormerPlusPlus':
                    frame0 = frame0 * 0.225 / 0.5 # approximate scaling so as to match the input range of FlowFormerPlusPlus
                for i in range(1,len(input)):
                    frame = F.interpolate(input[i], scale_factor=1.0, mode='nearest')
                    if args.optflow_model == 'FlowFormerPlusPlus':
                        frame = frame * 0.225 / 0.5 # approximate scaling so as to match the input range of FlowFormerPlusPlus
                    if 'Basic' in args.model_type:
                        flow = optflow_model(frame, frame0)[-1]
                    else:
                        flow = optflow_model(frame0, frame)[-1]
                    flow = F.interpolate(flow/1.0, size=(input[0].size(2), input[0].size(3)), mode='bilinear', align_corners=True)
                    optflow.append(flow)
                if args.add_depth_inputs: 
                    output = model(input, optflow=optflow, depth=input_depth)
                else:
                    output = model(input, optflow=optflow)
            elif args.add_depth_inputs: 
                output = model(input, depth=input_depth)
            else: 
                output = model(input)
            output = F.log_softmax(output, dim=1)

            metrics, metric_dict = get_metrics(output, mask, args.metric_fns, args)

            # get centroid prediction error
            err_rc, err_rb, err_lc, err_lb, pres_gt, pres, c_gt, c_pred = centroid_error(torch.exp(output), mask, args)
            centroid_pred_err_rt.append(err_rc)
            centroid_pred_err_rb.append(err_rb)
            centroid_pred_err_lt.append(err_lc)
            centroid_pred_err_lb.append(err_lb)
            centroid_pres_err_rt.append(pres_gt[0]^pres[0])
            centroid_pres_err_rb.append(pres_gt[1]^pres[1])
            centroid_pres_err_lt.append(pres_gt[2]^pres[2])
            centroid_pres_err_lb.append(pres_gt[3]^pres[3])

            batch_time.update(time.time() - batch_time_start)
            if step % args.save_output_freq == 0:
                # video_idx = step // (args.num_frames_per_video - args.num_input_frames + 1)
                # file_idx = step % (args.num_frames_per_video - args.num_input_frames + 1)
                # img_idx = video_idx*args.num_frames_per_video + file_idx + args.num_input_frames - 1 
                # disp_image = cv2.imread(str(file_names[img_idx]))
                disp_image = cv2.imread(str(file_names[step]))
                disp_image = cv2.resize(disp_image, (args.input_width, args.input_height))
                output_classes = output.data.cpu().numpy().argmax(axis=1)
                mask_array = output_classes[0]
                disp_image = mask_overlay(disp_image, (mask_array==1).astype(np.uint8), color=(255,1,0))
                disp_image = mask_overlay(disp_image, (mask_array==2).astype(np.uint8), color=(255,255,1))
                disp_image = mask_overlay(disp_image, (mask_array==3).astype(np.uint8), color=(0,1,255))
                disp_image = mask_overlay(disp_image, (mask_array==4).astype(np.uint8), color=(0,255,255))
                disp_image = draw_plus(disp_image, [c_gt[0][0],c_gt[1][0]], color=(0,255,0))
                disp_image = draw_plus(disp_image, [c_gt[0][1],c_gt[1][1]], color=(0,255,0))
                disp_image = draw_plus(disp_image, [c_gt[2][0],c_gt[3][0]], color=(0,255,0))
                disp_image = draw_plus(disp_image, [c_gt[4][0],c_gt[5][0]], color=(0,255,0))
                disp_image = draw_plus(disp_image, [c_gt[4][1],c_gt[5][1]], color=(0,255,0))
                disp_image = draw_plus(disp_image, [c_gt[6][0],c_gt[7][0]], color=(0,255,0))
                disp_image = draw_plus(disp_image, [c_pred[0][0],c_pred[1][0]], color=(255,255,255))
                disp_image = draw_plus(disp_image, [c_pred[0][1],c_pred[1][1]], color=(255,255,255))
                disp_image = draw_plus(disp_image, [c_pred[2][0],c_pred[3][0]], color=(255,255,255))
                disp_image = draw_plus(disp_image, [c_pred[4][0],c_pred[5][0]], color=(255,255,255))
                disp_image = draw_plus(disp_image, [c_pred[4][1],c_pred[5][1]], color=(255,255,255))
                disp_image = draw_plus(disp_image, [c_pred[6][0],c_pred[7][0]], color=(255,255,255))
                cv2.imwrite(str(args.output_dir / f'{step}.png'), disp_image)
            idx = 0
            for i, metric_fn in enumerate(args.metric_fns):
                for cls in range(1,args.num_classes):
                    progress_meter_list[idx+2].update(metrics[i][cls-1], input[0].size(0))
                    idx += 1
                # progress_meter_list[i+2].update(metric_dict['metric_'+metric_fn], input[0].size(0))
            if step % args.print_freq == 0:
                progress.display(step, logger=logger)
            step += 1
            data_time_start = time.time()
    
    # compute average detection accuracy
    logger.info(f'Avg. Centroid Detection Accuracy Right Tip: {(1.0-np.mean(centroid_pres_err_rt))*100}')
    logger.info(f'Avg. Centroid Detection Accuracy Right Base: {(1.0-np.mean(centroid_pres_err_rb))*100}')
    logger.info(f'Avg. Centroid Detection Accuracy Left Tip: {(1.0-np.mean(centroid_pres_err_lt))*100}')
    logger.info(f'Avg. Centroid Detection Accuracy Left Base: {(1.0-np.mean(centroid_pres_err_lb))*100}')
    logger.info(f'Std. Centroid Detection Error Right Tip: {np.std(centroid_pres_err_rt)*100}')
    logger.info(f'Std. Centroid Detection Error Right Base: {np.std(centroid_pres_err_rb)*100}')
    logger.info(f'Std. Centroid Detection Error Left Tip: {np.std(centroid_pres_err_lt)*100}')
    logger.info(f'Std. Centroid Detection Error Left Base: {np.std(centroid_pres_err_lb)*100}')

    # compute average centroid error; ignoring nans
    centroid_pred_err_rt = [x for x in centroid_pred_err_rt if not math.isnan(x)]
    centroid_pred_err_rb = [x for x in centroid_pred_err_rb if not math.isnan(x)]
    centroid_pred_err_lt = [x for x in centroid_pred_err_lt if not math.isnan(x)]
    centroid_pred_err_lb = [x for x in centroid_pred_err_lb if not math.isnan(x)]

    logger.info(f'Avg. Centroid Prediction Error Right Tip: {np.mean(centroid_pred_err_rt)} +/- {np.std(centroid_pred_err_rt)}')
    logger.info(f'Avg. Centroid Prediction Error Right Base: {np.mean(centroid_pred_err_rb)} +/- {np.std(centroid_pred_err_rb)}')
    logger.info(f'Avg. Centroid Prediction Error Left Tip: {np.mean(centroid_pred_err_lt)} +/- {np.std(centroid_pred_err_lt)}')
    logger.info(f'Avg. Centroid Prediction Error Left Base: {np.mean(centroid_pred_err_lb)} +/- {np.std(centroid_pred_err_lb)}')
    
    # compute average metrics
    idx = 0
    for i, metric_fn in enumerate(args.metric_fns):
        for cls in range(1,args.num_classes):
            logger.info(f"Avg. {metric_fn} for class {cls}: {progress_meter_list[idx].avg}")
            idx += 1
    # logger.info(f"Metrics: {metrics}")
    # logger.info(f"Avg. Metrics: {metric_dict}")
    return 

def main_worker(args):
    args.mode = 'testing'
    args.data_dir = Path(args.data_dir)
    args.log_dir = Path(os.path.join(args.expt_savedir, args.expt_name, 'logs'))
    args.output_dir = Path(os.path.join(args.expt_savedir, args.expt_name, 'outputs'))
    for dir in [args.log_dir, args.output_dir]:
        if not dir.is_dir():
            print(f"Creating {dir.resolve()} if non-existent")
            dir.mkdir(parents=True, exist_ok=True) 
    
    # set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(os.path.join(args.log_dir, "log.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Set seed
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Seed set to {seed}")

    # get test dataloader
    if args.dataset=='MICCAI2017':
        test_file_names, _ = get_MICCAI2017_dataset_filenames(args)
    elif args.dataset=='JIGSAWS':
        test_file_names, _ = get_JIGSAWS_dataset_filenames(args)
    elif args.dataset=='MICCAI2015':
        test_file_names, _ = get_MICCAI2015_dataset_filenames(args)
    elif args.dataset=='custom':
        test_file_names, _ = get_custom_dataset_filenames(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    # print(test_file_names)
    _, test_dataloader = get_data_loader(args)

    # set up optical flow model if needed
    if args.add_optflow_inputs:
        if args.optflow_model=='RAFT':
            from torchvision.models.optical_flow import raft_large
            optflow_model = raft_large(pretrained=True, progress=False)
            if torch.cuda.is_available():
                if torch.cuda.device_count() > 1:
                    optflow_model = nn.DataParallel(optflow_model)
                optflow_model = optflow_model.cuda()
        elif args.optflow_model=='FlowFormerPlusPlus':
            sys.path.append('./models/optical_flow/flowformerplusplus')
            sys.path.append('./models/optical_flow/flowformerplusplus/PerCostFormer3')
            from models.optical_flow.flowformerplusplus.ffpp_cfg_things import get_cfg
            from models.optical_flow.flowformerplusplus import build_flowformer
            cfg = get_cfg()
            optflow_model = build_flowformer(cfg)
            state_dict = torch.load(args.load_wts_optflow_model)
            new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
            optflow_model.load_state_dict(new_state_dict)
            if torch.cuda.is_available():
                if torch.cuda.device_count() > 1:
                    optflow_model = nn.DataParallel(optflow_model)
                optflow_model = optflow_model.cuda()
        else: 
            raise SystemError('GPU device not found! Not configured to train/test.')
        optflow_model.eval()
        logger.info(f"{args.optflow_model} optical flow model loaded")
    else: 
        optflow_model = None

    # set up model 
    model = get_model(args)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.cuda(); cudnn.benchmark = True
    else: 
        raise SystemError('GPU device not found! Not configured to train/test.')
    
    # load pre-trained weights if needed
    model, _, load_flag = load_model_weights(model, args.load_wts_model, args.model_type)
    if load_flag:
        logger.info("Model weights loaded from {}".format(args.load_wts_model))
    else: 
        logger.info("No model weights loaded")
    
    test(test_dataloader, model, args, test_file_names, logger, writer=None, optflow_model=optflow_model)
    return


if __name__ == '__main__':
    main()
