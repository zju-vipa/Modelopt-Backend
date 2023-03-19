from torch.utils.data import DataLoader
from torchvision import transforms
from loaders.datasets import ImageDataset


def _get_train_set(data_path):
    return ImageDataset(image_dir=data_path,
                        transform=transforms.Compose([
                            transforms.Resize((32, 32)),
                            transforms.RandomCrop(32, padding=2),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ]))


def _get_test_set(data_path):
    return ImageDataset(image_dir=data_path,
                        transform=transforms.Compose([
                            transforms.Resize((32, 32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ]))


def load_images(data_path, data_type=None):
    assert data_type is None or data_type in ['train', 'test']
    if data_type == 'train':
        data_set = _get_train_set(data_path)
    else:
        data_set = _get_test_set(data_path)

    data_loader = DataLoader(dataset=data_set,
                             batch_size=128,
                             num_workers=4,
                             shuffle=True)

    return data_loader
