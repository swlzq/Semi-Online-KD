from torchvision import transforms
from torchvision import datasets
import torch


def get_dataset(data_name, data_path):
    """
    Get dataset according to data name and data path.
    """
    transform_train, transform_test = data_transform(data_name)
    if data_name.lower() == 'cifar100':
        train_dataset = datasets.CIFAR100(data_path, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(data_path, train=False, download=True, transform=transform_test)
    else:
        raise NotImplementedError(f'No considering {data_name}')
    return train_dataset, test_dataset


def get_dataloader(data_name, data_path, batch_size):
    train_dataset, test_dataset = get_dataset(data_name, data_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def data_transform(data_name):
    transform_train, transform_test = None, None
    if data_name.lower().startswith('cifar'):
        transform_train = transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        transform_test = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    else:
        assert False
    return transform_train, transform_test
