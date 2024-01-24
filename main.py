import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging as log
from model import *
from data import *
from plots import plot_res
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.view(inputs.size(0), -1).to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

def print_gradients(grad1, grad2, grad3):
    print('Gradients:', grad3)

if __name__ == '__main__':
    # test()
    print(f'device = {device}')

    train_loader, valid_loader, test_loader = load_data()

    
    input_size = 28 * 28  # MNIST图像大小
    hidden_sizes = [128, 64, 32, 16]  # 5 层MLP
    # hidden_sizes = [28 * 28, 1024, 1024, 2048, 512, 512, 256, 256, 128, 128, 128, 64, 64, 64, 32, 32, 32, 16, 16, 16, 12]  # 22 层MLP
    # hidden_sizes = [28 * 28, 1024, 1024, 512, 512] + [256, 512, 1024, 512, 256] * 5 + [64, 64, 64, 32, 32, 32, 16, 16, 12, 12] # 41 层 MLP

    output_size = 10  # 类别

    mlp_model = MLP(input_size, hidden_sizes, output_size, BN=True).to(device)
    # mlp_model = Residual_MLP(input_size, hidden_sizes, output_size, BN=True).to(device)
    print(mlp_model)
    total_params = count_parameters(mlp_model)
    print(f"Total parameters: {total_params}")

    # hook_handle = mlp_model.model[0].register_backward_hook(print_gradients)  # hook 看梯度

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp_model.parameters(), lr=1e-3)
    # optimizer = optim.SGD(mlp_model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)

    epochs = 30
    val_acc_max = 0
    loss_all = []
    train_acc_all = []
    val_acc_all = []
    
    for epoch in range(epochs):
        mlp_model.train()
        torch.set_grad_enabled(True)
        loss_iter = []
        for inputs, labels in train_loader:
            inputs, labels = inputs.view(inputs.size(0), -1).to(device), labels.to(device)  # 将输入图像展平
            optimizer.zero_grad()
            outputs = mlp_model(inputs)
            loss = criterion(outputs, labels)
            loss_iter.append(loss.item())
            loss.backward()
            optimizer.step()
        loss_all.append(np.mean(loss_iter))

        train_accuracy = calculate_accuracy(mlp_model, train_loader, device)
        train_acc_all.append(train_accuracy * 100)
        val_accuracy = calculate_accuracy(mlp_model, valid_loader, device)
        val_acc_all.append(val_accuracy * 100)
        if val_accuracy > val_acc_max:
            val_acc_max = val_accuracy
            # torch.save(mlp_model.state_dict(), 'save_model/model.pth')
            with open('save_model/model.pkl', 'wb') as f:
                pickle.dump(mlp_model, f)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {np.mean(loss_iter) : .6f}, '
              f'train_acc: {train_accuracy * 100:.2f}%, val_acc: {val_accuracy * 100:.2f}%')

        loss_iter = []
        mlp_model.eval()
        torch.set_grad_enabled(False)
        for inputs, labels in valid_loader:
            inputs, labels = inputs.view(inputs.size(0), -1).to(device), labels.to(device)  # 将输入图像展平
            outputs = mlp_model(inputs)
            loss = criterion(outputs, labels)
            loss_iter.append(loss.item())
        # scheduler.step(np.mean(loss_iter))  # 学习率调整
            
    print("Training complete.")

    # mlp_model.load_state_dict(torch.load('save_model/model.pth'))
    with open('save_model/model.pkl', 'rb') as f:
        mlp_model = loaded_a = pickle.load(f)
    torch.set_grad_enabled(False)
    test_acc = calculate_accuracy(mlp_model, test_loader, device)
    print(f'test_acc: {test_acc * 100:.2f}%')

    plot_res(np.arange(epochs), loss_all, train_acc_all, val_acc_all, test_acc * 100)