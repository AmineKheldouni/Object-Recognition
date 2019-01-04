import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
import torchvision
from torchvision import models
import matplotlib.pyplot as plt
from model import *
from cnn_finetune import make_model

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='RecVis A3 training script')
    parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                        help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
    parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01')
    parser.add_argument('--momentum', type=float, default=0.7, metavar='M',
                        help='SGD momentum (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                        help='folder where experiment outputs are located.')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)

    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    # Data initialization and loading
    from data import data_transforms

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/train_images',
                             transform=data_transforms['train']),
        batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/val_images',
                             transform=data_transforms['val']),
        batch_size=args.batch_size, shuffle=False, num_workers=1)

    ##### ResNet34 model #####
    model = models.resnet34(pretrained=True)
    model.fc.out_features = 20

    #### FineTuned ResNet50 model with FC layers (Not working well) ####
    # model = FineTuneModel(num_classes = 20)
    ### Other implementation of ResNet50
    # def make_classifier(in_features, num_classes):
    #     return nn.Sequential(
    #         nn.Linear(in_features, 128),
    #         nn.ReLU(inplace=True),
    #         nn.Linear(128, 128),
    #         nn.ReLU(inplace=True),
    #         nn.Linear(128, num_classes),
    #     )
    # model = make_model('resnet50', num_classes=20, pretrained=True, input_size=(128, 128), classifier_factory=make_classifier)

    if use_cuda:
        print('Using GPU')
        model.cuda()
    else:
        print('Using CPU')

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # Scheduler for learning rate decrease each 10 epochs
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (0.5)**(epoch//5))

    # Lists for plotting learning curves
    learningCurve = []
    learningCurveVal = []

    def train(epoch):
        model.train()
        meanLoss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data.item()))
            meanLoss += loss.data.item()
        return meanLoss/len(train_loader)

    def validation():
        model.eval()
        validation_loss = 0
        correct = 0
        for data, target in val_loader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            # sum up batch loss
            criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
            validation_loss += criterion(output, target).data.item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        validation_loss /= len(val_loader.dataset)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            validation_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))
        return validation_loss

    for epoch in range(1, args.epochs + 1):
        # Print actual learning rate
        for param_group in optimizer.param_groups:
            print("lr = ", param_group['lr'])
        scheduler.step()
        l = train(epoch)
        learningCurve.append(l)
        lv = validation()
        learningCurveVal.append(lv)
        model_file = args.experiment + '/model_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_file)
        print('\nSaved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file')
    print(learningCurve)
    print(learningCurveVal)
    plt.plot(range(1, args.epochs + 1), learningCurve, color='blue', label='Training')
    plt.plot(range(1, args.epochs + 1), learningCurveVal, color='green', label='Validation')
    plt.xlabel("epochs", fontsize=14)
    plt.ylabel("Cross entropy loss", fontsize=14)
    plt.legend()
    plt.savefig('learningcurve.png')
    plt.show()
