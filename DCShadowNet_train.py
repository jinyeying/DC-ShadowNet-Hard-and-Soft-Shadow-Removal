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

        self.use_ch_loss = args.use_ch_loss
        self.use_pecp_loss = args.use_pecp_loss
        self.use_smooth_loss = args.use_smooth_loss 
        
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
        print("# datasetpath : ", self.datasetpath)
    
    def build_model(self):
        """ DataLoader """
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.img_size + 30, self.img_size+30)),
            transforms.RandomCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.trainA = ImageFolder(os.path.join('dataset', self.datasetpath, 'trainA'), train_transform)
        self.trainB = ImageFolder(os.path.join('dataset', self.datasetpath, 'trainB'), train_transform)
        if self.use_ch_loss:
            self.trainC = ImageFolder(os.path.join('dataset', self.datasetpath, 'train_A_intr2d_light'), train_transform) ##offline load physics ch_norm

        self.testA = ImageFolder(os.path.join('dataset', self.datasetpath, 'testA'), test_transform)
        self.testB = ImageFolder(os.path.join('dataset', self.datasetpath, 'testB'), test_transform)
        if self.use_ch_loss:
            self.testC = ImageFolder(os.path.join('dataset', self.datasetpath, 'test_A_intr2d_light'), test_transform)    ##offline load physics ch_norm
        
        self.trainA_loader = DataLoader(self.trainA, batch_size=self.batch_size, shuffle=True)
        self.trainB_loader = DataLoader(self.trainB, batch_size=self.batch_size, shuffle=True)
        if self.use_ch_loss:
            self.trainC_loader = DataLoader(self.trainC, batch_size=self.batch_size, shuffle=False)

        self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False)
        self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False)
        if self.use_ch_loss:
            self.testC_loader = DataLoader(self.testC, batch_size=1, shuffle=False)

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

        """ Trainer """
        self.G_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.D_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters(), self.disLA.parameters(), self.disLB.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.Rho_clipper = RhoClipper(0, 1)

    def train(self):
        self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

        start_iter = 1
        if self.resume:
            model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))
            if not len(model_list) == 0:
                model_list.sort()
                start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.result_dir, self.dataset, 'model'), start_iter)
                print(" [*] Load SUCCESS")
                if self.decay_flag and start_iter > (self.iteration // 2):
                    self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
                    self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)

        # training loop
        print('training start !')
        start_time = time.time()
        for step in range(start_iter, self.iteration + 1):
            if self.decay_flag and step > (self.iteration // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))

            try:
                real_A, _ = trainA_iter.next()
            except:
                trainA_iter = iter(self.trainA_loader)
                real_A, _ = trainA_iter.next()

            try:
                real_B, _ = trainB_iter.next()
            except:
                trainB_iter = iter(self.trainB_loader)
                real_B, _ = trainB_iter.next()

            if self.use_ch_loss:
                try:
                    real_C, _ = trainC_iter.next()
                except:
                    trainC_iter = iter(self.trainC_loader)
                    real_C, _ = trainC_iter.next()

            real_A, real_B = real_A.to(self.device), real_B.to(self.device)
            if self.use_ch_loss:
                real_C = real_C.to(self.device)

            # Update D
            self.D_optim.zero_grad()

            fake_A2B, _, _ = self.genA2B(real_A)
            fake_B2A, _, _ = self.genB2A(real_B)

            real_GA_logit, real_GA_Dom_logit, _ = self.disGA(real_A)
            real_LA_logit, real_LA_Dom_logit, _ = self.disLA(real_A)
            real_GB_logit, real_GB_Dom_logit, _ = self.disGB(real_B)
            real_LB_logit, real_LB_Dom_logit, _ = self.disLB(real_B)

            fake_GA_logit, fake_GA_Dom_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_Dom_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_Dom_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_Dom_logit, _ = self.disLB(fake_A2B)

            D_ad_loss_GA     = self.MSE_loss(real_GA_logit,     torch.ones_like(real_GA_logit).to(self.device))     + self.MSE_loss(fake_GA_logit,     torch.zeros_like(fake_GA_logit).to(self.device))
            D_ad_Dom_loss_GA = self.MSE_loss(real_GA_Dom_logit, torch.ones_like(real_GA_Dom_logit).to(self.device)) + self.MSE_loss(fake_GA_Dom_logit, torch.zeros_like(fake_GA_Dom_logit).to(self.device))
            D_ad_loss_LA     = self.MSE_loss(real_LA_logit,     torch.ones_like(real_LA_logit).to(self.device))     + self.MSE_loss(fake_LA_logit,     torch.zeros_like(fake_LA_logit).to(self.device))
            D_ad_Dom_loss_LA = self.MSE_loss(real_LA_Dom_logit, torch.ones_like(real_LA_Dom_logit).to(self.device)) + self.MSE_loss(fake_LA_Dom_logit, torch.zeros_like(fake_LA_Dom_logit).to(self.device))
            
            D_ad_loss_GB     = self.MSE_loss(real_GB_logit,     torch.ones_like(real_GB_logit).to(self.device))     + self.MSE_loss(fake_GB_logit,     torch.zeros_like(fake_GB_logit).to(self.device))
            D_ad_Dom_loss_GB = self.MSE_loss(real_GB_Dom_logit, torch.ones_like(real_GB_Dom_logit).to(self.device)) + self.MSE_loss(fake_GB_Dom_logit, torch.zeros_like(fake_GB_Dom_logit).to(self.device))
            D_ad_loss_LB     = self.MSE_loss(real_LB_logit,     torch.ones_like(real_LB_logit).to(self.device))     + self.MSE_loss(fake_LB_logit,     torch.zeros_like(fake_LB_logit).to(self.device))
            D_ad_Dom_loss_LB = self.MSE_loss(real_LB_Dom_logit, torch.ones_like(real_LB_Dom_logit).to(self.device)) + self.MSE_loss(fake_LB_Dom_logit, torch.zeros_like(fake_LB_Dom_logit).to(self.device))

            D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_Dom_loss_GA + D_ad_loss_LA + D_ad_Dom_loss_LA)
            D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_Dom_loss_GB + D_ad_loss_LB + D_ad_Dom_loss_LB)

            Discriminator_loss = D_loss_A + D_loss_B
            Discriminator_loss.backward()
            self.D_optim.step()

            # Update G
            self.G_optim.zero_grad()
     
            fake_A2B, fake_A2B_Dom_logit, _ = self.genA2B(real_A)       ##yy: fake_B = netG_A2B(real_A), Gb(a)
            fake_B2A, fake_B2A_Dom_logit, _ = self.genB2A(real_B)       ##yy: fake_A = netG_B2A(real_B), Ga(b)

            fake_A2B2A, _, _ = self.genB2A(fake_A2B)                    ##yy: recovered_A = netG_B2A(fake_B), Ga(Gb(a))
            fake_B2A2B, _, _ = self.genA2B(fake_B2A)                    ##yy: recovered_B = netG_A2B(fake_A), Gb(Ga(b))

            fake_A2A, fake_A2A_Dom_logit, _ = self.genB2A(real_A)       #yy: G_B2A(A) should equal A if real_A is fed, same_A = netG_B2A(real_A), same_A is fake_A2A
            fake_B2B, fake_B2B_Dom_logit, _ = self.genA2B(real_B)       #yy: G_A2B(B) should equal B if real_B is fed, same_B = netG_A2B(real_B), same_B is fake_B2B

            fake_GA_logit, fake_GA_Dom_logit, _ = self.disGA(fake_B2A)  #Da(Ga(b))_global
            fake_LA_logit, fake_LA_Dom_logit, _ = self.disLA(fake_B2A)  #Da(Ga(b))_local
            fake_GB_logit, fake_GB_Dom_logit, _ = self.disGB(fake_A2B)  #Db(Gb(a))_global
            fake_LB_logit, fake_LB_Dom_logit, _ = self.disLB(fake_A2B)  #Db(Gb(a))_local

            G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))             ##yy: log(Da(Ga(b))), loss_GAN_B2A = criterion_GAN(pred_fake, target_real), global D
            G_ad_Dom_loss_GA = self.MSE_loss(fake_GA_Dom_logit, torch.ones_like(fake_GA_Dom_logit).to(self.device)) ##yy: G Dom
            G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))             ##yy: log(Da(Ga(b))), local D 
            G_ad_Dom_loss_LA = self.MSE_loss(fake_LA_Dom_logit, torch.ones_like(fake_LA_Dom_logit).to(self.device)) ##yy: L Dom
            
            G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))             ##yy: log(Db(Gb(a))), loss_GAN_A2B = criterion_GAN(pred_fake, target_real), global D
            G_ad_Dom_loss_GB = self.MSE_loss(fake_GB_Dom_logit, torch.ones_like(fake_GB_Dom_logit).to(self.device))
            G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))             ##yy: log(Db(Gb(a))), local D
            G_ad_Dom_loss_LB = self.MSE_loss(fake_LB_Dom_logit, torch.ones_like(fake_LB_Dom_logit).to(self.device))

            G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)           #yy: ||Ga(Gb(a))-a||1, loss_cycle_ABA
            G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)           #yy: ||Gb(Ga(b))-b||1, loss_cycle_BAB

            G_identity_loss_A = self.L1_loss(fake_A2A, real_A)          #yy: ||Ga(a)-a||1, loss_identity_A 
            G_identity_loss_B = self.L1_loss(fake_B2B, real_B)          #yy: ||Gb(b)-b||1, loss_identity_B
            ##Binary Domain-Classifier 
            G_dom_loss_A = self.BCE_loss(fake_B2A_Dom_logit, torch.ones_like(fake_B2A_Dom_logit).to(self.device)) + self.BCE_loss(fake_A2A_Dom_logit, torch.zeros_like(fake_A2A_Dom_logit).to(self.device)) ##fake_A, 1; same_A(fake_A2A) 0
            G_dom_loss_B = self.BCE_loss(fake_A2B_Dom_logit, torch.ones_like(fake_A2B_Dom_logit).to(self.device)) + self.BCE_loss(fake_B2B_Dom_logit, torch.zeros_like(fake_B2B_Dom_logit).to(self.device)) ##fake_B, 1; same_B(fake_B2B) 0
            
            if self.use_pecp_loss:
                selfpecpvgg_loss = PerceptualLossVgg16(None,
                                            [0],
                                            weights=[1.0], 
                                            indices=[22])                ##yy: Checkpoint/Model for pecp_vgg loss, ImageNet, layer 22
                loss_selfpecp = selfpecpvgg_loss(fake_A2B, real_A)
            
            if self.use_smooth_loss:
                gen_mask    = softmask_generator(real_A, fake_A2B)
                loss_smooth = smooth_loss_masked(fake_A2B, gen_mask)

            if self.use_ch_loss:
                fake_A2B_ = (fake_A2B+1.)/2.
                ch_z = fake_A2B_/ fake_A2B_.sum(dim=1, keepdim=True).clamp(min=1e-8) ##yy: for fake_A2B do chromaticity operation c/(r+g+b) to get ch_z
                ch_z    = 2*ch_z-1
                ch_norm = real_C                                                     ##yy: offline load ch_norm 
                loss_ch =  self.L1_loss(ch_z, ch_norm)

            G_loss_A = self.adv_weight * (G_ad_loss_GA + G_ad_Dom_loss_GA + G_ad_loss_LA + G_ad_Dom_loss_LA) + self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + self.dom_weight * G_dom_loss_A
            G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_Dom_loss_GB + G_ad_loss_LB + G_ad_Dom_loss_LB) + self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + self.dom_weight * G_dom_loss_B

            Generator_loss = G_loss_A + G_loss_B

            if self.use_ch_loss == True:
                Generator_loss = Generator_loss + loss_ch
            if self.use_pecp_loss == True:
                Generator_loss = Generator_loss + loss_selfpecp
            if self.use_smooth_loss == True:
                Generator_loss = Generator_loss + loss_smooth

            Generator_loss.backward()
            self.G_optim.step()

            self.genA2B.apply(self.Rho_clipper)
            self.genB2A.apply(self.Rho_clipper)

            print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss))
            with torch.no_grad():
                if step % self.print_freq == 0:
                    train_sample_num = 5
                    test_sample_num = 5
                    A2B = np.zeros((self.img_size * 4, 0, 3))

                    self.genA2B.eval(), self.genB2A.eval(), self.disGA.eval(), self.disGB.eval(), self.disLA.eval(), self.disLB.eval()
                    
                    for _ in range(train_sample_num):
                        try:
                            real_A, _ = trainA_iter.next()
                        except:
                            trainA_iter = iter(self.trainA_loader)
                            real_A, _ = trainA_iter.next()

                        try:
                            real_B, _ = trainB_iter.next()
                        except:
                            trainB_iter = iter(self.trainB_loader)
                            real_B, _ = trainB_iter.next()

                        real_A, real_B = real_A.to(self.device), real_B.to(self.device)

                        fake_A2B, _, _ = self.genA2B(real_A)
                        fake_B2A, _, _ = self.genB2A(real_B)

                        fake_A2B2A, _, _ = self.genB2A(fake_A2B)           ##yy: recovered_A = netG_B2A(fake_B), Ga(Gb(a))
                        fake_B2A2B, _, _ = self.genA2B(fake_B2A)           ##yy: recovered_B = netG_A2B(fake_A), Gb(Ga(b))

                        fake_A2A, _, _ = self.genB2A(real_A)               #yy: G_B2A(A) should equal A if real A is fed, same_A = netG_B2A(real_A), same_A(fake_A2A)
                        fake_B2B, _, _ = self.genA2B(real_B)               #yy: G_A2B(B) should equal B if real B is fed, same_B = netG_A2B(real_B), same_B(fake_B2B)


                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),                                                                                                                             
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),   
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                    for _ in range(test_sample_num):
                        try:
                            real_A, _ = testA_iter.next()
                        except:
                            testA_iter = iter(self.testA_loader)
                            real_A, _ = testA_iter.next()

                        try:
                            real_B, _ = testB_iter.next()
                        except:
                            testB_iter = iter(self.testB_loader)
                            real_B, _ = testB_iter.next()

                        if self.use_ch_loss:
                            try:
                                real_C_test, _ = testC_iter.next()
                            except:
                                testC_iter = iter(self.testC_loader)
                                real_C_test, _ = testC_iter.next()

                        real_A, real_B = real_A.to(self.device), real_B.to(self.device)
                        if self.use_ch_loss:
                            real_C_test = real_C_test.to(self.device)

                        fake_A2B, _, _ = self.genA2B(real_A)
                        fake_B2A, _, _ = self.genB2A(real_B)

                        fake_A2B2A, _, _ = self.genB2A(fake_A2B)
                        fake_B2A2B, _, _ = self.genA2B(fake_B2A)

                        fake_A2A, _, _ = self.genB2A(real_A)
                        fake_B2B, _, _ = self.genA2B(real_B)

                        if self.use_smooth_loss == True:
                            gen_mask = softmask_generator(real_A, fake_A2B)
                            A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                                       RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                                       RGB2BGR(tensor2numpy(denorm(gen_mask[0]))),
                                                                       RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)
                        if self.use_ch_loss == True:
                            fake_A2B_ = (fake_A2B+1.)/2.
                            ch_z = fake_A2B_/ fake_A2B_.sum(dim=1, keepdim=True).clamp(min=1e-8)
                            ch_z_test = 2*ch_z-1
                            ch_norm_test = real_C_test
                            A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                                       RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                                       RGB2BGR(tensor2numpy(denorm(ch_norm_test[0]))),
                                                                       RGB2BGR(tensor2numpy(denorm(ch_z_test[0])))), 0)), 1)

                    cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'train_img', 'A2B_%07d.png' % step), A2B * 255.0)
                    self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

                if step % self.save_freq == 0:
                    self.save(os.path.join(self.result_dir, self.dataset, 'model'), step)

                if step % 1000 == 0:
                    params = {}
                    params['genA2B'] = self.genA2B.state_dict()
                    params['genB2A'] = self.genB2A.state_dict()
                    params['disGA'] = self.disGA.state_dict()
                    params['disGB'] = self.disGB.state_dict()
                    params['disLA'] = self.disLA.state_dict()
                    params['disLB'] = self.disLB.state_dict()
                    torch.save(params, os.path.join(self.result_dir, self.dataset + '_params_latest.pt'))

    def save(self, dir, step):
        params = {}
        params['genA2B'] = self.genA2B.state_dict()
        params['genB2A'] = self.genB2A.state_dict()
        params['disGA'] = self.disGA.state_dict()
        params['disGB'] = self.disGB.state_dict()
        params['disLA'] = self.disLA.state_dict()
        params['disLB'] = self.disLB.state_dict()
        torch.save(params, os.path.join(dir, self.dataset + '_params_%07d.pt' % step))

    def load(self, dir, step):
        params = torch.load(os.path.join(dir, self.dataset + '_params_%07d.pt' % step))
        self.genA2B.load_state_dict(params['genA2B'])
        self.genB2A.load_state_dict(params['genB2A'])
        self.disGA.load_state_dict(params['disGA'])
        self.disGB.load_state_dict(params['disGB'])
        self.disLA.load_state_dict(params['disLA'])
        self.disLB.load_state_dict(params['disLB'])
