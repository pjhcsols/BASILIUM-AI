import argparse
import torch

class TestOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
    
    def initialize(self):
        self.parser.add_argument('--name', type=str, default='demo', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')        
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32], help="Supported data type i.e. 8, 16, 32 bit")
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')

        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=512, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        self.parser.add_argument('--input_path', type=str, default='/home/woo/Desktop/job/project/VITON/test/datasets') 
        self.parser.add_argument('--output_path', type = str, default = '/home/woo/Desktop/job/project/VITON/test/results')

        self.parser.add_argument('--resize_or_crop', type=str, default='scale_width', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')        
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation') 
        self.parser.add_argument('--nThreads', default=1, type=int, help='# threads for loading data')                
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        self.parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        self.parser.add_argument('--original_height', type = int, default = 256)
        self.parser.add_argument('--original_width', type = int, default = 192)

        self.isTrain = False

        VitonOptions.initialize(self.parser)
        SROptions.initialize(self.parser)

        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        if self.opt.gpu_ids != '-1':
            self.opt.gpu_ids = list(map(int, self.opt.gpu_ids.split(',')))
            torch.cuda.set_device(self.opt.gpu_ids[0])
            self.opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            if self.opt.device == 'cpu':
                print('cuda is not available, use cpu')

        # args = vars(self.opt)

        # print('------------ Options -------------')
        # for k, v in sorted(args.items()):
        #     print('%s: %s' % (str(k), str(v)))
        # print('-------------- End ----------------')

        return self.opt


class VitonOptions:
    def initialize(parser):
        parser.add_argument('--warp_model_checkpoint', type=str, default='/home/woo/Desktop/job/project/VITON/test/model_zoo/PFAFN_warp_epoch_101.pth', help='load the pretrained model from the specified location')
        parser.add_argument('--gen_model_checkpoint', type=str, default='/home/woo/Desktop/job/project/VITON/test/model_zoo/PFAFN_gen_epoch_101.pth', help='load the pretrained model from the specified location')


class SROptions:
    def initialize(parser):
        parser.add_argument('--task', type = str, default = 'real_sr')
        parser.add_argument('--scale', type = int, default = 4)
        parser.add_argument('--noise', type = int, default = 0)
        parser.add_argument('--jpeg', type = int, default = 40)
        parser.add_argument('--training_patch_size', type = int, default = 128)
        parser.add_argument('--large_model', type = int, default = 1)
        parser.add_argument('--sr_model_checkpoint', type = str, default = '/home/woo/Desktop/job/project/VITON/test/model_zoo/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth')
        parser.add_argument('--tile', type = bool, default = None)
        parser.add_argument('--tile_overlap', type = int, default = 32)
        parser.add_argument('--denoise50_model_checkpoint', type = str, default = '/home/woo/Desktop/job/project/VITON/test/model_zoo/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth')
        parser.add_argument('--denoise25_model_checkpoint', type = str, default = '/home/woo/Desktop/job/project/VITON/test/model_zoo/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth')
        parser.add_argument('--denoise15_model_checkpoint', type = str, default = '/home/woo/Desktop/job/project/VITON/test/model_zoo/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth')
