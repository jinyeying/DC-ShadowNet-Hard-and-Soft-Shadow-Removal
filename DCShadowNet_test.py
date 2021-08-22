import time, itertools
from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from networks import *
from utils_loss import *
from glob import glob
from PIL import Image

class DCShadowNet(object) :
    def __init__(self, args):        
        self.model_name = 'DCShadowNet'

        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.datasetpath = args.datasetpath

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.dom_weight = args.dom_weight

        if args.use_ch_loss == True:
            self.ch_weight = args.ch_weight
        if args.use_pecp_loss == True:
            self.pecp_weight = args.pecp_weight
        if args.use_smooth_loss == True:
            self.smooth_weight = args.smooth_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_h = args.img_h
        self.img_w = args.img_w
        self.img_ch = args.img_ch

        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume

        if torch.backends.cudnn.enabled and self.benchmark_flag:
            print('set benchmark !')
            torch.backends.cudnn.benchmark = True

        print()

        print("##### Information #####")
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ DataLoader """
        self.test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
      
        self.testA = ImageFolder(os.path.join('dataset', self.datasetpath, 'testA'), self.test_transform)
        self.testB = ImageFolder(os.path.join('dataset', self.datasetpath, 'testB'), self.test_transform)
        self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False)
        self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False)

        """ Define Generator, Discriminator """
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=True).to(self.device)
        self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=True).to(self.device)
        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)
        self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)

        """ Define Loss """
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

        """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
        self.Rho_clipper = RhoClipper(0, 1)

    def load(self, dir, step):
        params = torch.load(os.path.join(dir, self.dataset + '_params_%07d.pt' % step))
        self.genA2B.load_state_dict(params['genA2B'])
        self.genB2A.load_state_dict(params['genB2A'])
        self.disGA.load_state_dict(params['disGA'])
        self.disGB.load_state_dict(params['disGB'])
        self.disLA.load_state_dict(params['disLA'])
        self.disLB.load_state_dict(params['disLB'])

    def test(self):
        to_pil = transforms.ToPILImage()
        model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))
        if not len(model_list) == 0:
            model_list.sort()
            print('model_list',model_list)
            iter = int(model_list[-1].split('_')[-1].split('.')[0])
            for i in range(-1,0,1):
                self.load(os.path.join(self.result_dir, self.dataset, 'model'), iter)
                print(" [*] Load SUCCESS")

                self.genA2B.eval(), self.genB2A.eval()
                
                path_real_A=os.path.join(self.result_dir, self.dataset, str(iter)+'/inputA')
                if not os.path.exists(path_real_A):
                    os.makedirs(path_real_A)
                
                path_fake_B=os.path.join(self.result_dir, self.dataset, str(iter)+'/outputB')
                if not os.path.exists(path_fake_B):
                    os.makedirs(path_fake_B)

                path_realAfakeB=os.path.join(self.result_dir, self.dataset, str(iter)+'/inputA_outputB')
                if not os.path.exists(path_realAfakeB):
                    os.makedirs(path_realAfakeB)

                for n, (real_A, _) in enumerate(self.testA_loader):
                    real_A = real_A.to(self.device)

                    fake_A2B, _, _ = self.genA2B(real_A)

                    A_real = RGB2BGR(tensor2numpy(denorm(real_A[0])))
                    B_fake = RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))

                    A2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                          RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))), 1)

                    cv2.imwrite(os.path.join(self.result_dir, self.dataset, str(iter)+'/inputA',  '%d.png' % (n + 1)), A_real * 255.0) 
                    cv2.imwrite(os.path.join(self.result_dir, self.dataset, str(iter)+'/outputB',  '%d.png' % (n + 1)), B_fake * 255.0)
                    cv2.imwrite(os.path.join(self.result_dir, self.dataset, str(iter)+'/inputA_outputB',  '%d.png' % (n + 1)), A2B * 255.0) 
