from dataloaders.datasets import own_dataset
from torch.utils.data import DataLoader


def make_data_loader(args, **kwargs):
    if args.dataset == 'datasets':
        train_set = own_dataset.Segmentation(args, split='train')
        val_set = own_dataset.Segmentation(args, split='val')
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **kwargs)
        return train_loader, val_loader, args.num_classes

    else:
        raise NotImplementedError
