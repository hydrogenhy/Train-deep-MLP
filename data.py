from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def load_data():
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    transform = transforms.ToTensor()
    mnist_dataset  = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # 划分训练集和验证集
    train_size = int(0.8 * len(mnist_dataset))  # 80% 训练集
    valid_size = len(mnist_dataset) - train_size  # 20% 验证集
    train_dataset, valid_dataset = random_split(mnist_dataset, [train_size, valid_size])    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, valid_loader, test_loader