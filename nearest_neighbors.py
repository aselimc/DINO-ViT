import os
import random
import argparse
import torch
from pprint import pprint
from torchvision.transforms import *
from utils import check_dir, get_logger
from utils.weights import load_from_weights
from models.pretraining_backbone import ResNet18Backbone, DINOHead
from data.pretraining import DataReaderPlainImg
from torch.utils.data import DataLoader
from pretrain import MultiCropWrapper

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-init', type=str,
                        default="results/lr0.0005_bs64__local/models/ckpt_epoch9.pth")
    parser.add_argument("--size", type=int, default=256, help="size of the images to feed the network")
    parser.add_argument('--output-root', type=str, default='results')
    args = parser.parse_args()

    args.output_folder = check_dir(
        os.path.join(args.output_root, "nearest_neighbors",
                     args.weights_init.replace("/", "_").replace("models", "")))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):
    # model
    #raise NotImplementedError("TODO: build model and load weights snapshot")
    model = ResNet18Backbone(pretrained=True)
    model = MultiCropWrapper(model, DINOHead(
        512, 128, norm_last_layer=True,
    ))
    model = load_from_weights(model, args.weights_init)
    model = model.cuda()
    model.eval()
    # dataset
    val_transform = Compose([Resize(args.size), CenterCrop((args.size, args.size)), ToTensor()])
    #raise NotImplementedError("Load the validation dataset (crops), use the transform above.")
    val_data = DataReaderPlainImg("dataset/crops/images/256/val", transform=val_transform)
    val_loader = DataLoader(val_data, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    # choose/sample which images you want to compute the NNs of.
    # You can try different ones and pick the most interesting ones.
    query_indices = [12, 22, 61, 85, 98]
    nns = []
    nns_big = []
    logger = get_logger(args.logs_folder, "nearest_neighbors")
    for idx, img in enumerate(val_loader):
        if idx not in query_indices:
            continue
        print("Computing NNs for sample {}".format(idx))
        closest_idx, closest_dist = find_nn(model, img.to(torch.device(0)), val_loader, 5)
        c_idx = [i.item() for i in closest_idx]
        c_dist = [round(i.item(),6) for i in closest_dist]
        logger.info(f"The index of closest NNs for the {idx}th image are {c_idx}, and respective distances are {c_dist}")
        #raise NotImplementedError("TODO: retrieve the original NN images, save them and log the results.")
        nns = [val_data[i] for i in c_idx]
        concatenated = torch.cat(nns, 2)
        # functional.to_pil_image(concatenated).save(f"results/nearest_neighbors/nn_{idx}.png")
        nns_big.append(concatenated)
        nns = []
    all = torch.cat(nns_big, 1)
    functional.to_pil_image(all).save(f"results/nearest_neighbors/all.png")




def find_nn(model, query_img, loader, k):
    """
    Find the k nearest neighbors (NNs) of a query image, in the feature space of the specified mode.
    Args:
         model: the model for computing the features
         query_img: the image of which to find the NNs
         loader: the loader for the dataset in which to look for the NNs
         k: the number of NNs to retrieve
    Returns:
        closest_idx: the indices of the NNs in the dataset, for retrieving the images
        closest_dist: the L2 distance of each NN to the features of the query image
    """
    with torch.no_grad():
        dist = []
        query_out = model(query_img)
        for data in loader:
            data = data.to(torch.device(0))
            prediction = model(data)
            distance = torch.linalg.norm(query_out-prediction, ord=2)
            dist.append(distance.item())
        dist = torch.Tensor(dist)
        # First one being the image itself
        closest_idx = torch.argsort(dist)[:k+1]
        closest_dist = dist[closest_idx]

    # raise NotImplementedError("TODO: nearest neighbors retrieval")
    return closest_idx, closest_dist


if __name__ == '__main__':
    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args) 
