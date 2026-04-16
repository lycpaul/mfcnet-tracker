"""
Configuration file stating all the args for toolpose segmentation task
"""

def train_config_parser(parser):
    # dataset related arguments
    parser.add_argument('--data_dir', type=str, default='/mnt/iMVR/shuojue/data/surgpose/', 
                        help='Path to data directory. Default: /mnt/iMVR/shuojue/data/surgpose/')
    parser.add_argument('--dataset', type=str, default='SurgPose', choices=['MICCAI2015', 'MICCAI2017', 'JIGSAWS', 'SurgPose'],
                        help='Dataset name. Default: SurgPose')
    parser.add_argument('--fold_index', type=int, default=-1, choices=[-1,0,1,2,3], 
                        help='Fold index for cross validation. Default: -1, no cross validation')
    parser.add_argument('--prediction_task', type=str, default='surgpose_segmentation_single',
                        help='Prediction task. Default: surgpose_segmentation_single')
    parser.add_argument('--mode', type=str, default='training', choices=['training', 'testing'], 
                        help='Mode of operation. Default: training')
    parser.add_argument('--num_frames_per_video', type=int, default=501, 
                        help='Number of frames per video/folder in the dataset. Default: 501')
    parser.add_argument('--sparse_view_ratio', type=int, default=1, 
                        help='sparse view ratio for testing. Default: 1')
    # I/O related arguments
    parser.add_argument('--expt_savedir', type=str, default='./results', 
                        help='Path to save experiment results. Default: ./results')
    parser.add_argument('--expt_name', type=str, default='SurgPose_test',
                        help='Experiment name. Default: SurgPose_test')
    parser.add_argument('--print_freq', type=int, default=250, 
                        help='Print frequency. Default: 250')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save frequency. Default: 5')
    parser.add_argument('--debug', type=bool, default=False, help='Debug mode')

    # optimizer related arguments   
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size. Default: 4')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of workers for dataloader. Default: 12')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of classes (incl. background). Default: 6')
    parser.add_argument('--metric_fns', type=str, nargs='+', default=['iou', 'dice', 'coco'], choices=['iou', 'dice', 'coco'], 
                        help='List of metric functions. Default: iou, dice, coco')
    parser.add_argument('--loss_fns', type=str, nargs='+', default=['nll', 'soft_jaccard'], choices=['mse', 'nll', 'soft_jaccard'],  
                        help='List of loss functions. Default: nll, soft_jaccard')
    parser.add_argument('--loss_wts', type=float, nargs='+', default=[0.7, 0.3], 
                        help='List of loss weights. Default: 0.7 0.3')
    parser.add_argument('--lr', type=float, default=3e-5, 
                        help='Learning rate. Default: 3e-5')
    parser.add_argument('--scheduler', type=str, default='StepDecay', choices=['StepDecay', 'Constant'], 
                        help='Learning rate scheduler. Default: StepDecay at halfway point')
    parser.add_argument('--num_epochs', type=int, default=20, 
                        help='Number of epochs. Default: 20')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Seed for random number generator. Default: 42')
    parser.add_argument('--resume', type=bool, default=False, 
                        help='Resume training')
    parser.add_argument('--starting_epoch', type=int, default=0, 
                        help='Starting epoch. Default: 0')
    parser.add_argument('--class_weights', type=float, nargs='+', default=[1,1000,1000,1000,1000,1000],
                        help='Class weights for NLL loss function. Default: [1,1000,1000,1000,1000,1000]')
    
    # model related arguments
    parser.add_argument('--model_type', type=str, default='FCN', 
                        choices=['TernausNet11', 'TernausNet16', 'TAPNet11', 'TAPNet16', 'DeepLab_v3', 'FCN', 'HRNet', 'SegFormer'], 
                        help='Model name')
    parser.add_argument('--pretrained', type=bool, default=True, 
                        help='Use pre-trained weights. Default: True')
    parser.add_argument('--load_wts_model', type=str, default=None, 
                        help='Path to model weights. Default: None')
    parser.add_argument('--input_height', type=int, default=480, help='NN input image height')
    parser.add_argument('--input_width', type=int, default=640, help='NN input image width')
    parser.add_argument('--add_optflow_inputs', type=bool, default=False, help='Add optical flow inputs')
    parser.add_argument('--optflow_dir', type=str, default=None, 
                        choices=['optflows_unflow', 'optflows_raft'])
    parser.add_argument('--update_attmaps', type=bool, default=False, help='Update attention maps')
    return parser

def test_config_parser(parser):
    # dataset related arguments
    parser.add_argument('--data_dir', type=str, default='/mnt/iMVR/shuojue/data/surgpose/', 
                        help='Path to data directory. Default: /mnt/iMVR/shuojue/data/surgpose/')
    parser.add_argument('--dataset', type=str, default='SurgPose', choices=['MICCAI2015', 'MICCAI2017', 'JIGSAWS', 'SurgPose'],
                        help='Dataset name. Default: SurgPose')
    parser.add_argument('--prediction_task', type=str, default='surgpose_segmentation_single',
                        help='Prediction task. Default: surgpose_segmentation_single')
    parser.add_argument('--num_frames_per_video', type=int, default=501, 
                        help='Number of frames per video/folder in the dataset. Default: 501')

    
    # I/O related arguments
    parser.add_argument('--expt_savedir', type=str, default='./results', 
                        help='Path to save experiment results. Default: ./results')
    parser.add_argument('--expt_name', type=str, default='SurgPose_test_full',
                        help='Experiment name. Default: SurgPose_test')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save frequency. Default: 5')
    parser.add_argument('--print_freq', type=int, default=250, 
                        help='Print frequency. Default: 250')
    parser.add_argument('--save_output_freq', type=int, default=1,
                        help='Save output frequency. Default: 10')
    parser.add_argument('--skip_output_images', action='store_true',
                        help='Skip saving output visualization images during inference.')

    # optimizer related arguments   
    parser.add_argument('--num_workers', type=int, default=12, help='Number of workers for dataloader. Default: 12')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of classes (incl. background). Default: 6')
    parser.add_argument('--metric_fns', type=str, nargs='+', default=['iou', 'dice', 'coco'], choices=['iou', 'dice', 'coco'], 
                        help='List of metric functions. Default: iou, dice, coco')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Seed for random number generator. Default: 42')
    parser.add_argument('--resume', type=bool, default=False, 
                        help='Resume training. Default: False')
    
    # model related arguments
    parser.add_argument('--model_type', type=str, default='FCN', 
                        choices=['TernausNet11', 'TernausNet16', 'TAPNet11', 'TAPNet16', 'DeepLab_v3', 'FCN', 'HRNet', 'SegFormer'], 
                        help='Model name')
    parser.add_argument('--pretrained', type=bool, default=False, 
                        help='Use pre-trained weights. Default: False')
    # parser.add_argument('--load_wts_model', type=str, default='results/SurgPose_test_full/ckpts/model_020.pth', 
    #                     help='Path to model weights. Default: None')
    parser.add_argument('--input_height', type=int, default=480, help='NN input image height')
    parser.add_argument('--input_width', type=int, default=640, help='NN input image width')
    parser.add_argument('--add_optflow_inputs', type=bool, default=False, help='Add optical flow inputs')
    parser.add_argument('--optflow_dir', type=str, default=None, 
                        choices=['optflows_unflow', 'optflows_raft'])
    parser.add_argument('--update_attmaps', type=bool, default=False, help='Update attention maps')
    parser.add_argument('--pth_file_name', type=str, default=None, help='Name of the pth file')
    
    return parser

