import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


###########   preparation   ############
def prepare_dataset(args, split, transform, preprocess):
    imsize = args.imsize
    if transform is not None:
        image_transform = transform
    elif args.CONFIG_NAME.find('CelebA') != -1:
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])
    else:
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])
    # train dataset
    from datasets import TextImgDataset as Dataset
    dataset = Dataset(split=split, transform=image_transform, args=args, preprocess=preprocess)
    return dataset


def prepare_datasets(args, transform, preprocess):
    # train dataset
    train_dataset = prepare_dataset(args, split='train', transform=transform, preprocess= preprocess)
    # test dataset
    val_dataset = prepare_dataset(args, split='val', transform=transform, preprocess= preprocess)
    return train_dataset, val_dataset


def prepare_dataloaders(args, transform=None, preprocess= None):
    batch_size = args.batch_size
    num_workers = args.num_workers
    train_dataset, valid_dataset = prepare_datasets(args, transform, preprocess)
    # train dataloader
    if args.multi_gpus == True:
        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, drop_last=True,
            num_workers=num_workers, sampler=train_sampler)
    else:
        train_sampler = None
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, drop_last=True,
            num_workers=num_workers, shuffle='True')
    # valid dataloader
    # if args.multi_gpus == True:
        # valid_sampler = DistributedSampler(valid_dataset)
        # valid_dataloader = torch.utils.data.DataLoader(
        #     valid_dataset, batch_size=batch_size, drop_last=True,
        #     num_workers=num_workers, sampler=valid_sampler, shuffle='False')
    # else:
    valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, drop_last=True,
            num_workers=num_workers, shuffle = False)
    return train_dataloader, valid_dataloader, \
           train_dataset, valid_dataset, train_sampler



################################################################################################
# prepare clip
################################################################################################
# import clip
#
# def load_clip():
#     clip_model, preprocess = clip.load('ViT-B/32')
#     clip_model = clip_model.cuda()
#     for param in clip_model.parameters():
#         param.requires_grad = False
#     return clip_model
