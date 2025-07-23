from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import torch
import torchvision
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from utils import *
import bagnets.pytorchnet
from PyTorch_Sobel.pytorch_sobel import *
from skimage import feature as skif
import torch.nn.functional as F
from DropSAM import SAM
from DropSAM import enable_running_stats, disable_running_stats
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

class Features(nn.Module):
    def __init__(self, net_layers):
        super(Features, self).__init__()
        self.net_layer_0 = nn.Sequential(net_layers[0])
        self.net_layer_1 = nn.Sequential(net_layers[1])
        self.net_layer_2 = nn.Sequential(net_layers[2])
        self.net_layer_3 = nn.Sequential(net_layers[3])
        self.net_layer_4 = nn.Sequential(*net_layers[4])
        self.net_layer_5 = nn.Sequential(*net_layers[5])
        self.net_layer_6 = nn.Sequential(*net_layers[6])
        self.net_layer_7 = nn.Sequential(*net_layers[7])


    def forward(self, x):
        x = self.net_layer_0(x)
        x = self.net_layer_1(x)
        x = self.net_layer_2(x)
        x = self.net_layer_3(x)
        x = self.net_layer_4(x)
        x1 = self.net_layer_5(x)
        x2 = self.net_layer_6(x1)
        x3 = self.net_layer_7(x2)
        return x1, x2, x3
    


class Edge_Estimator(nn.Module):
    def __init__(self):
        super(Edge_Estimator, self).__init__()
        
        self.mu = nn.Sequential(
                                nn.ReLU(),
                                nn.Conv2d(2048, 128, kernel_size=1),
                                nn.ReLU(),
                                nn.Conv2d(128, 2048, kernel_size=1))
        self.log_var = nn.Sequential(
                        nn.ReLU(),
                        nn.Conv2d(2048, 128, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(128, 2048, kernel_size=1))
        self.Up = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(2048, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 2048, kernel_size=1),
            nn.ReLU(),
            nn.ConvTranspose2d(2048, 1, 33, stride=8))

    def forward(self, x):
        mu = self.mu(x)
        log_var = self.log_var(x)
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return self.Up(z)
        
        
   
    
class Color_Texture_Estimator(nn.Module):
    def __init__(self):
        super(Color_Texture_Estimator, self).__init__()
        
        self.mu = nn.Sequential(
                                nn.AdaptiveAvgPool2d((1,1)),
                                Squeeze(),
                                nn.ReLU(),
                                nn.Linear(2048, 128),
                                nn.ReLU(),
                                nn.Linear(128, 354))
        self.log_var = nn.Sequential(
                        nn.AdaptiveAvgPool2d((1,1)),
                        Squeeze(),
                        nn.ReLU(),
                        nn.Linear(2048, 128),
                        nn.ReLU(),
                        nn.Linear(128, 354))

    def forward(self, x):
        mu = self.mu(x)
        log_var = self.log_var(x)
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
        
 
    
class XAI_Wrapper(nn.Module):
    def __init__(self, net_layers, fc_dim=2048, num_class=7):
        super().__init__()
        self.Features = Features(net_layers)
        
        
        self.Branch_EG = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.Branch_CT = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.Decoder_EG = Edge_Estimator()
        self.Decoder_CT = Color_Texture_Estimator()
        
        
        self.Att_EG = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.Sigmoid())
        self.Att_CT = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.Sigmoid())
        self.GlobalPooling = nn.AdaptiveAvgPool2d((1,1))
        self.Classifier = nn.Sequential(nn.Linear(fc_dim*2, num_class))

    def forward(self, x):
        _, _, x = self.Features(x)
        
        x_eg = self.Branch_EG(x)
        x_ct = self.Branch_CT(x)
        
        eg = self.Decoder_EG(x_eg)
        ct = self.Decoder_CT(x_ct)
        

        x_eg_select = torch.mul(x_eg, self.Att_EG(x_eg))
        x_ct_select = torch.mul(x_ct, self.Att_CT(x_ct))
        
        
        
        x = torch.cat([x_eg_select, x_ct_select],axis=1)
        
        
        x = self.GlobalPooling(x).view(x.size(0), -1)
        x = self.Classifier(x)

        return x, eg, ct
    
def denormalisation(feature):
    mean = torch.tensor([0.5, 0.5, 0.5]).reshape(3,1,1).to(feature.device)
    std = torch.tensor([0.5, 0.5, 0.5]).reshape(3,1,1).to(feature.device)
    return feature*std+mean


def tensor2im(feature):
    if feature.ndim==2:
        feature=feature.unsqueeze(0)
        feature=feature.repeat(3,1,1)
        
    f_max,_ = feature.view(feature.size(0),feature.size(1),-1).max(2)
    f_min,_ = feature.view(feature.size(0),feature.size(1),-1).min(2)
    f_max = f_max.unsqueeze(-1).unsqueeze(-1)
    f_min = f_min.unsqueeze(-1).unsqueeze(-1)
    feature = (feature-f_min)/(f_max-f_min+1e-6)
    return feature


def rgb2bgr(im):
    return im[:,[2,1,0],:,:]


class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.view(x.size(0), -1)

class Edge_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.Sobel = GradLayer()
        
    def forward(self, x):
        x= tensor2im(x)
        x = self.Sobel(x)
        return x
    
class  Color_Texture_Generator:
    def lbp_histogram(self,image,P=8,R=1,method = 'nri_uniform'):
        '''
        image: shape is N*M 
        '''
        lbp = skif.local_binary_pattern(image, P,R, method) # lbp.shape is equal image.shape
        # cv2.imwrite("lbp.png",lbp)
        max_bins = int(lbp.max() + 1) # max_bins is related P
        hist,_= np.histogram(lbp,  density=True, bins=max_bins, range=(0, max_bins))
        return hist
    
    def lbp_histogram_tensor(self,x):
        features = []
        for i in range(x.shape[0]):
            image=x[i]
            b_h = self.lbp_histogram(image[:,:,0]) # b channel
            g_h = self.lbp_histogram(image[:,:,1]) # g channel
            r_h = self.lbp_histogram(image[:,:,2]) # r channel
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            y_h = self.lbp_histogram(image[:,:,0]) # y channel
            cb_h = self.lbp_histogram(image[:,:,1]) # cb channel
            cr_h = self.lbp_histogram(image[:,:,2]) # cr channel
            
            feature = np.concatenate((b_h, g_h, r_h, y_h,cb_h,cr_h))
            feature = torch.from_numpy(feature.astype('float32'))
            feature = feature.unsqueeze(0)
            features.append(feature)
            
        return torch.cat(features,axis=0)

    def process(self, x):
        x = tensor2im(rgb2bgr(x))
        x = x.permute(0,2,3,1)
        x = (torch.round(x.clamp(0.0,1.0)*255)).cpu().numpy().astype(np.uint8)
        x = self.lbp_histogram_tensor(x)
        
        var = x.std(0).unsqueeze(0)
        avg = x.mean(0).unsqueeze(0)
        x = (x-avg)/(var+1e-6)
        
        return x


def test(net, criterion, batch_size, data_path=''):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    total = 0
    idx = 0
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = "cpu"
    
    net.to(device)

    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(225),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_set = torchvision.datasets.ImageFolder(root=data_path,
                                               transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs), Variable(targets)
        with torch.no_grad():
            outputs,_,_ = net(inputs)

        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)

        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if batch_idx % 50 == 0  or batch_idx == (test_loader.__len__()-1):
            print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))

    test_acc = 100. * float(correct) / total

    test_loss = test_loss / (idx + 1)

    return test_acc, test_loss

def inference(net, criterion, batch_size, data_path=''):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    total = 0
    idx = 0
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = "cpu"
    net.to(device)

    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(225),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_set = torchvision.datasets.ImageFolder(root=data_path,
                                               transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    score_list = []
    target_list = []
    pred_list = []
    
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs), Variable(targets)
        with torch.no_grad():
            outputs,_,_ = net(inputs)

        score_list.append(outputs.softmax(dim=1).data.cpu())
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        pred_list.append(predicted.data.cpu().unsqueeze(0))
        target_list.append(targets.data.cpu().unsqueeze(0))

        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if batch_idx % 50 == 0 or batch_idx == (test_loader.__len__()-1):
            print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))

    pred_list = torch.cat(pred_list, axis=-1).squeeze().numpy()
    target_list = torch.cat(target_list, axis=-1).squeeze().numpy()
    score_list = torch.cat(score_list, axis=0).squeeze().numpy()
    
    accuracy = accuracy_score(pred_list, target_list)*100
    f1_micro = f1_score(target_list,pred_list,average='micro')
    f1_macro = f1_score(target_list,pred_list,average='macro')
    
    auc_micro = roc_auc_score(target_list, score_list, multi_class='ovr',average='micro')
    auc_macro = roc_auc_score(target_list, score_list, multi_class='ovr',average='macro')
    
    
    
    test_acc = 100. * float(correct) / total

    test_loss = test_loss / (idx + 1)
    print("Test Accuracy: {}%".format(test_acc))

    return accuracy, f1_micro, f1_macro, auc_micro,auc_macro




def train(nb_epoch, batch_size, num_class, store_name, lr=0.002, data_path='', start_epoch=0):
    # setup output
    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)
    
  

    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(225, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.ImageFolder(root=os.path.join(data_path,'train'), transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    
    net = bagnets.pytorchnet.bagnet33(pretrained=True)
    
    net_layers = list(net.children())
    net_layers = net_layers[0:8]
    net = XAI_Wrapper(net_layers, fc_dim=2048, num_class=num_class)
    

    # GPU
    device = torch.device("cuda")
    net.to(device)

    CELoss = nn.CrossEntropyLoss()
    lossfunc = nn.KLDivLoss( reduction='sum')
    base_optimizer = torch.optim.SGD
    optimizer = SAM(net.parameters(), base_optimizer, adaptive=False, lr=lr, momentum=0.9, weight_decay=5e-4)
    
    
    max_val_acc = 0
    EG = Edge_Generator()
    if use_cuda:
        EG.to(device)
    CT = Color_Texture_Generator()

    for epoch in range(start_epoch, nb_epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        idx = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            idx = batch_idx
            inputs_ct = CT.process(inputs).to(device)
            inputs_edge = EG(inputs.to(device)).clone().detach()

            
            
            if inputs.shape[0] < batch_size:
                continue
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            
            inputs, targets = Variable(inputs), Variable(targets)
            inputs_ct, inputs_edge = Variable(inputs_ct), Variable(inputs_edge)

 
            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr)

            enable_running_stats(net)
            optimizer.zero_grad()
            outputs, outputs_eg, outputs_ct = net(inputs)
            outputs_eg = outputs_eg.view(outputs_eg.size(0),-1)
            inputs_edge = inputs_edge.view(inputs_edge.size(0),-1)
            loss = CELoss(outputs, targets) + \
                torch.log(lossfunc(F.log_softmax(outputs_eg, dim=1), F.softmax(inputs_edge, dim=1))+1)+\
                torch.log(lossfunc(F.log_softmax(outputs_ct, dim=1), F.softmax(inputs_ct, dim=1))+1)
            loss.backward()
            optimizer.first_step(0.5, zero_grad=True)
            
            disable_running_stats(net)
            outputs, outputs_eg, outputs_ct = net(inputs)
            outputs_eg = outputs_eg.view(outputs_eg.size(0),-1)
            loss = CELoss(outputs, targets) + \
                torch.log(lossfunc(F.log_softmax(outputs_eg, dim=1), F.softmax(inputs_edge, dim=1))+1)+\
                torch.log(lossfunc(F.log_softmax(outputs_ct, dim=1), F.softmax(inputs_ct, dim=1))+1)
            loss.backward()
            optimizer.second_step(zero_grad=True)
            

            #  training log
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            train_loss += loss.item() 
            

            if batch_idx % 50 == 0 or batch_idx == (trainloader.__len__()-1):
                print(
                    'Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    batch_idx, train_loss / (batch_idx + 1),
                    100. * float(correct) / total, correct, total))

        train_acc = 100. * float(correct) / total
        train_loss = train_loss / (idx + 1)
        with open(exp_dir + '/results_train.txt', 'a') as file:
            file.write(
                'Iteration %d | train_acc = %.5f | train_loss = %.5f |\n' % (
                epoch, train_acc, train_loss))


        val_acc, val_loss = test(net, CELoss, batch_size, os.path.join(data_path, 'validation'))
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            net.cpu()
            torch.save(net, './' + store_name + '/model.pth')
            net.to(device)
        with open(exp_dir + '/results_test.txt', 'a') as file:
            file.write('Iteration %d, test_acc = %.5f, test_loss = %.6f\n' % (
            epoch, val_acc, val_loss))
    
    
    
    trained_model = torch.load('./' + store_name + '/model.pth')
    accuracy, f1_micro, f1_macro, auc_micro, auc_macro = inference(trained_model, CELoss, batch_size, os.path.join(data_path, 'test'))
    with open(exp_dir + '/results_test.txt', 'a') as file:
        file.write('Inference Results: Accuracy = %.5f, F1_micro = %.5f, F1_macro = %.5f, Auc_micro = %.5f, Auc_macro = %.5f \n' % (
        accuracy, f1_micro, f1_macro, auc_micro, auc_macro))
    
    
            
        

if __name__ == '__main__':
    seed_everything(0)
    
    results_path = 'results'
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    
    task_result_path = os.path.join(results_path, 'classification')
    if not os.path.exists(task_result_path):
        os.mkdir(task_result_path)
    

        

    data_path = ''
    dataset_organizing_style = '311'
    
    experiment_result_path = \
                        os.path.join(task_result_path,os.path.basename(__file__).replace('.py','')) + \
                        '_' + dataset_organizing_style

   
    lr = 0.002
    train(nb_epoch=100,             
            batch_size=16,
            num_class=7,
            lr = lr,      
            store_name= experiment_result_path,     
            data_path=data_path,          
            start_epoch=0)         
