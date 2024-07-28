import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import os
import sys
import argparse
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from pytorch_nndct.apis import torch_quantizer

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

classes = ('Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)                 
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )


        
    def forward(self, x):
        # We break down the x = self.features(x) to minor modules
        x = self.features[0](x)  #conv
        x = self.features[1](x)  #relu
        x = self.features[2](x)  #maxpool
      
        x = self.features[3](x)  #conv
        x = self.features[4](x)  #relu
        x = self.features[5](x)  #maxpool
       
        x = self.features[6](x)  #conv
        x = self.features[7](x)  #relu
      
        x = self.features[8](x)  #conv
        x = self.features[9](x)  #relu      

        x = self.features[10](x) #conv
        x = self.features[11](x) #relu
        x = self.features[12](x) #maxpool
     
        
        x = self.avgpool(x)
      
        x = x.view(x.size(0), 256 * 6 * 6)    

        x = self.classifier[0](x) #drop
        x = self.classifier[1](x) #linear           
        x = self.classifier[2](x) #relu
        x = self.classifier[3](x) #drop       
        x = self.classifier[4](x) #linear
        x = self.classifier[5](x) #relu
        x = self.classifier[6](x) #linear

      
        return x


if __name__ == '__main__':

    # Initialize parser and setting the hyper parameters
    parser = argparse.ArgumentParser(description="AlexNet Network with CIFAR10 Dataset")
    parser.add_argument('--epochs', default=7, type=int, help="Achieves a testing accuracy of 85.07% with only 7 epochs")
    parser.add_argument('--batch_size', default=4, type=int, help="Batch size of 4 is a good choice for Training")
    parser.add_argument('--lr', default=0.001, type=float, help="Initial learning rate")
    parser.add_argument('--weights', default='alexnet_cifar_50.pkl', help="The path to the saved weights")
    parser.add_argument('--dataset', default='build/dataset', help="The path to the train and test datasets")
    parser.add_argument('--phase', default='calib', choices=['train', 'calib', 'test'])
    args = parser.parse_args()
    print(args)

    # CIFAR10 dataset and dataloader declaration
    transform = transforms.Compose([
        transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    batchsize = args.batch_size

    # Override batchsize if in test mode and deployment of xmodel
    if args.phase == 'test':
        batchsize = 1

    train_data = torchvision.datasets.CIFAR10(root=args.dataset, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root=args.dataset, train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=False, num_workers=0)

    # To use the genuine AlexNet with 1000 classes as the Network Model
    model = AlexNet(num_classes=10)
    #print(model)

    # Device Initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"\n A {device} is assigned for processing! \n")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    if args.phase == 'train':
        # Training
        EPOCHS = args.epochs
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            start_time = time.time()
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                output = model(inputs)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 1000 == 999:    # print every 1000 mini-batches
                    #print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                    #print('Time:',time_taken)
                    running_loss = 0.0

                                
            if not os.path.isdir("models"):
                os.mkdir("models")
            torch.save(model.state_dict(), f'models/alexnet_cifar_{epoch+1}.pkl')    

        print('Training of AlexNet has been finished')
    
    elif args.phase == 'calib':

        ###### Load model #######
        if args.weights is not None:
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(args.weights))
            else:
                model.load_state_dict(torch.load(args.weights, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['alexnet'], 'Pretrained_weights'))

        ###### Quantization #######
        quant_mode = args.phase
        quant_model = 'build/quant_model'
        rand_in = torch.randn([batchsize, 3, 256, 256])  # RGB 32*32 input images (CIFAR10)
        quantizer = torch_quantizer(quant_mode, model, (rand_in), output_dir=quant_model)
        quantized_model = quantizer.quant_model

        ###### Evaluation #######
        correct = 0
        total = 0

        quantized_model.eval()
        with torch.no_grad():
            for index, (data, target) in enumerate(tqdm(test_loader, desc='Data Progress')):
                image, target = data.to(device), target.to(device)
                outputs = quantized_model(image)
                
             

                _, predicted = torch.max(outputs.data, 1) # To find the index of max-probability for each output in the BATCH
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        print()
        print('Accuracy of the network on 10000 test images: %.2f %%' % (100 * correct / total))
        print("Evaluation in " + args.phase + " finished")

    elif args.phase == 'test':

        ###### Load model #######
        if args.weights is not None:
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(args.weights))
            else:
                model.load_state_dict(torch.load(args.weights, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['alexnet'], 'Pretrained_weights'))

        ###### Quantization #######
        quant_mode = args.phase
        quant_model = 'build/quant_model'
        rand_in = torch.randn([batchsize, 3, 32, 32])  # RGB 32*32 input images (CIFAR10)
        quantizer = torch_quantizer(quant_mode, model, (rand_in), output_dir=quant_model)
        quantized_model = quantizer.quant_model

        ###### Evaluation #######
        correct = 0
        total = 0

        model.eval()
        with torch.no_grad():
            for index, (data, target) in enumerate(tqdm(test_loader, desc='Data Progress')):
                image, target = data.to(device), target.to(device)
                outputs = model(image)
                
             

                _, predicted = torch.max(outputs.data, 1) # To find the index of max-probability for each output in the BATCH
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        print()
        print('Accuracy of the network on 10000 test images: %.2f %%' % (100 * correct / total))
        print("Evaluation in " + args.phase + " finished")

# Train:   python3 train_test_cifar10.py --phase 'train'
# Calib:    python3 train_test_cifar10.py --phase 'calib' --batch_size 100
# Test:    python3 train_test_cifar10.py --phase 'test'
