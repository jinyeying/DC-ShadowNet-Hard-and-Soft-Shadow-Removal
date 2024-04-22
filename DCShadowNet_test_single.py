import time, itertools
from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from networks import *
from utils_loss import *
from glob import glob
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

class DCShadowNet(object) :
    def __init__(self, args):        
        self.model_name = 'DCShadowNet'

        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.datasetpath = args.datasetpath
        self.ch = args.ch
        self.n_res = args.n_res
        self.img_size = args.img_size
        self.device = args.device

        print("##### Information #####")
        print("# dataset : ", self.dataset)
        print("# datasetpath : ", self.datasetpath)

    def build_model(self):
        self.test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
      
        self.testA = ImageFolder(os.path.join(self.datasetpath), self.test_transform)
        self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False)
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=True).to(self.device)

    def load(self, dir, step):
        params = torch.load(os.path.join(dir, self.dataset + '_params_%07d.pt' % step), map_location=torch.device(self.device))
        self.genA2B.load_state_dict(params['genA2B'])

    def test(self):
        model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))
        print(model_list)
        if not len(model_list) == 0:
            model_list.sort()
            iter = int(model_list[-1].split('_')[-1].split('.')[0])
            for i in range(-1,0,1):
                self.load(os.path.join(self.result_dir, self.dataset, 'model'), iter)
                print(" [*] Load SUCCESS")

                self.genA2B.eval()
                
                path_fakeB=os.path.join(self.result_dir, 'output')
                print('output saved in:', path_fakeB)
                if not os.path.exists(path_fakeB):
                    os.makedirs(path_fakeB)

                path_realAfakeB=os.path.join(self.result_dir, 'input_output')
                print('input_output saved in:', path_realAfakeB)
                if not os.path.exists(path_realAfakeB):
                    os.makedirs(path_realAfakeB)

                self.test_list = [os.path.splitext(f) for f in os.listdir(os.path.join(self.datasetpath)) if any(f.endswith(suffix) for suffix in IMG_EXTENSIONS)]
                for n, in_name in enumerate(self.test_list):
                    print('predicting: %d / %d' % (n + 1, len(self.test_list)))
                    img_name = in_name[0]
                    im_suf   = in_name[-1]
                    img = Image.open(os.path.join(self.datasetpath, img_name + im_suf)).convert('RGB')
                    
                    real_A = (self.test_transform(img).unsqueeze(0)).to(self.device)
                    
                    fake_A2B, _, _ = self.genA2B(real_A)
                    
                    A_real = RGB2BGR(tensor2numpy(denorm(real_A[0])))
                    B_fake = RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))
                    A2B = np.concatenate((A_real, B_fake), 1)

                    cv2.imwrite(os.path.join(path_fakeB,  '%s.png' % img_name), B_fake * 255.0)
                    cv2.imwrite(os.path.join(path_realAfakeB,'%s.png' % img_name), A2B * 255.0)
