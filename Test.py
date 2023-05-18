import torch
import timm
import glob
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms, ToTensor
import torchvision.transforms as T
import torch.optim as optim
import matplotlib.pyplot as plt 

device = 'cuda:0'
model = timm.create_model('xception',num_classes=2, pretrained=True).to(device)

test = torch.rand((1,3,167,167)).to(device)

class deepfakeDataset(Dataset):
    def __init__(self, path, train=None, transform=None):
        self.path = path
        if train == 'train':
            self.fake_path = path + '/train/fake'
            self.real_path = path + '/train/real'
        elif train == 'test':
            self.fake_path = path + '/test/fake'
            self.real_path = path + '/test/real'
        else:
            self.fake_path = path + '/val/fake'
            self.real_path = path + '/val/real'

        self.fake_list = glob.glob(self.fake_path+'/*.png')
        self.real_list = glob.glob(self.real_path+'/*.png')

        self.transform = transform

        self.data_list = self.fake_list + self.real_list
        self.class_list = [[1.,0.]] * len(self.fake_list) + [[0.,1.]] * len(self.real_list)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img_path = self.data_list[idx]
        label = torch.Tensor(self.class_list[idx]).to(device)
        img = Image.open(img_path)

        img = self.transform(img).to(device)
        return img, label

class Net(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        out = self.model(x)
        return out

loss_fn = nn.CrossEntropyLoss()

if __name__ == "__main__":
    transform = transforms.Compose(
        [
            T.Resize(228),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]
    )
    trainset = deepfakeDataset(path='./High Quality', train='train', transform=transform)
    trainloader = DataLoader(dataset=trainset,
                        batch_size=40,
                        shuffle=True,
                        drop_last=False)
    testset = deepfakeDataset(path='./High Quality', train='test', transform=transform)
    testloader = DataLoader(dataset=testset,
                        batch_size=40,
                        shuffle=True,
                        drop_last=False)
    valset = deepfakeDataset(path='./High Quality', train='val', transform=transform)
    valloader = DataLoader(dataset=valset,
                        batch_size=40,
                        shuffle=True,
                        drop_last=False)
    
    
    net = Net(model)
    net.to(device)
    opt = optim.Adam(net.parameters(), lr=3e-4)

    loss_arr = []
    plot_range = [i for i in range(60)]
    # train
    for epoch in range(20):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            opt.zero_grad()

            outputs = net(inputs)
            outputs = nn.functional.softmax(outputs, dim=1) #softmax
            loss = loss_fn(outputs, labels) #cross-entropy
            loss.backward()
            opt.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print(f'[ {epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                loss_arr.append(running_loss/10)
                running_loss = 0.0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i, data in enumerate(valloader, 0):
            inputs, labels = data
            outputs = net(inputs)
            outputs = nn.functional.softmax(outputs, dim=1)
            outputs = outputs.argmax(1)
            labels = labels.argmax(1)
            for i in range(len(labels)):
                if labels[i] == outputs[i]:
                    if labels[i] == 1:
                        tp += 1
                    else:
                        tn += 1
                else:
                    if labels[i] == 1:
                        fn += 1
                    else:
                        fp += 1
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = (2*precision*recall)/(precision+recall)
        print(f'acc = {accuracy:.3f}, prec = {precision:.3f}, rec = {recall:.3f}, f1 = {f1:.3f}')
    
    plt.plot(plot_range, loss_arr)
    plt.show()

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        outputs = net(inputs)
        outputs = nn.functional.softmax(outputs, dim=1)
        outputs = outputs.argmax(1)
        labels = labels.argmax(1)
        for i in range(len(labels)):
            if labels[i] == outputs[i]:
                if labels[i] == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if labels[i] == 1:
                    fn += 1
                else:
                    fp += 1
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = (2*precision*recall)/(precision+recall)
    print(f'acc = {accuracy:.3f}, prec = {precision:.3f}, rec = {recall:.3f}, f1 = {f1:.3f}')