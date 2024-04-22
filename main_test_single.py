from DCShadowNet_test_single import DCShadowNet
import argparse
from utils_loss import *

def parse_args():
    desc = "Pytorch implementation of DCShadowNet"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='test', help='[train / test]')
    parser.add_argument('--dataset', type=str, default='SRD', help='dataset_name')
    parser.add_argument('--datasetpath', type=str, default='./test_input/', help='dataset_path')
    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Set gpu mode; [cpu, cuda]')
    return check_args(parser.parse_args())

def check_args(args):
    check_folder(os.path.join(args.result_dir, args.dataset, 'model'))
    return args

def main():
    args = parse_args()
    if args is None:
      exit()

    gan = DCShadowNet(args)

    gan.build_model()

    if args.phase == 'test' :
        gan.test()
        print(" [*] Test finished!")

if __name__ == '__main__':
    main()