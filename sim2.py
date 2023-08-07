import torch
import torch.nn as nn
import torch.nn.functional as F # for relu
import torch.linalg
import torch.nn.init as init
from math import sqrt
from load_dataset import load_mnist
class training_set(torch.utils.data.Dataset):
    def __init__(self,X,Y):
        self.X = X                           # set data
        self.Y = Y                           # set lables

    def __len__(self):
        return len(self.X)                   # return length

    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]]    # return list of batch data [data, labels]

class ImplicitLayer(nn.Module):
    def __init__(self, input_features, width, output_features = 1, act = torch.tanh, sigma_w = 0.6, sigma_u = 1.0, tol = None, max_iter=50, outlayer=True):
        super().__init__()
        self.Flatten = nn.Flatten()
        # first layber
        self.input = nn.Linear(input_features, width, bias=False)
        init.normal_(self.input.weight, mean=0, std=sigma_u/sqrt(input_features)) # initialization of U

        # implicit layer
        self.implicit = nn.Linear(width, width, bias=False)
        init.normal_(self.implicit.weight, mean=0, std=sigma_w/sqrt(width)) # initialization of W

        # output layer
        self.output = nn.Linear(width, output_features, bias=False) # output layer, default initialization

        # init.normal_(self.output.weight, mean=0, std=1./sqrt(output_features)) # initialization of W
        # set hyper-parameters
        self.input_features = input_features
        self.width = width

        self.gamma = sigma_w
        self.sigma_u = sigma_u
        self.tol = tol
        self.max_iter = max_iter

        self.act = act
        self.outlayer = outlayer

    def forward(self, x):
        # first layer
        # x = self.Flatten(x)
        # x = F.relu(self.input(x))
        ux = self.input(x)
        # initialize output z to be zero
        z = torch.zeros_like(ux)
        self.iterations = 0

        while self.iterations < self.max_iter:
            z_next = self.act(self.implicit(z) + ux)
            self.err = torch.norm(z - z_next)
            z = z_next
            self.iterations += 1
            if self.tol is not None:
              if self.err < self.tol:
                  break
        if self.outlayer:
          return self.output(z)
        else:
          return z


criterion = nn.MSELoss()
# a generic function for running a single epoch (training or evaluation)
def epoch(loader, model, opt=None, monitor=None):
    total_loss, total_err, total_monitor = 0.,0.,0.
    model.eval() if opt is None else model.train()
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        X=X.reshape([len(X),-1])
        X = (X-X.mean(-1,keepdims=True))/X.std(-1,keepdims=True)
        yp = model(X)
        
        # loss = nn.CrossEntropyLoss()(yp,y)
        one_hot = y# torch.nn.functional.one_hot(y, yp.shape[1])
        loss = criterion(yp,one_hot.float())
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_err += (yp.max(dim=1)[1] != y.max(dim=1)[1]).sum().item()
        total_loss += loss.item() * X.shape[0]
        if monitor is not None:
            total_monitor += monitor(model)
    return total_err / len(loader.dataset), total_loss / len(loader.dataset), total_monitor / len(loader) 

# setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# import datasets and data loader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# need to create a method and input is name of the datasets, and batch_size
# and return the train and test loaders
train_data = datasets.MNIST(
    root = "data",
    train = True,
    download = True,
    transform = transforms.ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# choose class 0 and 1
# train_idx =[]
# for i in range(len(train_data.targets)):
#     if train_data.targets[i] <= 1:
#         train_idx.append(i)
        
# test_idx =[]
# for i in range(len(test_data.targets)):
#     if test_data.targets[i] <= 1:
#         test_idx.append(i)   

# train_data = torch.utils.data.Subset(train_data, train_idx)
# test_data = torch.utils.data.Subset(test_data, test_idx)

# choose a sampel with size=num
num = 500
train_idx = torch.randperm(len(train_data))[:num]
test_idx = torch.randperm(len(test_data))[:10000]

train_data = torch.utils.data.Subset(train_data, train_idx)
test_data = torch.utils.data.Subset(test_data, test_idx)

# Create data loader

train_dataloader = DataLoader(train_data, batch_size=len(train_data), shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=len(train_data), shuffle=True)
batch_size = 500
num_train = 500
act = torch.relu
sigma_w = 0.5
sigma_u = 1.0
width = 500
depth = 10
lr = 0.02


(train_image, train_label,
        valid_image, valid_label,
        test_image, test_label) =  load_mnist(num_train=num_train)
train_image =torch.from_numpy(train_image)
train_lanbel = torch.from_numpy(train_label)
test_image = torch.from_numpy(test_image)
test_label = torch.from_numpy(test_label)
train_image =  (train_image-train_image.mean(-1,keepdims=True))/train_image.std(-1,keepdims=True)
test_image =  (test_image-test_image.mean(-1,keepdims=True))/test_image.std(-1,keepdims=True)
train_data = training_set(train_image, train_label)
test_data = training_set(test_image,test_label)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

print(f"{len(train_data)} training data | {len(test_data)} test data | batch size {train_dataloader.batch_size}")

input_dim = len(test_data[0][0].reshape(-1))
# out_dim = len(torch.unique(test_data.targets))
out_dim = 2
print(f"Input dim {input_dim} | output dim {out_dim}")

# import optimizer
import torch.optim as optim
# need to create a method to take width as dataset dim, implicit layers, max iteration, output dim
def train(model,gamma, input_dim, width, out_dim, train_data, test_data, lr, max_iter=1000):
    # model = nn.Sequential(nn.Flatten(),
    #     nn.Linear(input_dim, width, bias=False),
    #     reluImplicitLayer(width, gamma = gamma, max_iter=100),
    #     nn.Linear(width, out_dim, bias=False)).to(device)
    # model = reluImplicitNet(input_dim, width, out_dim).to(device)

    opn = model.gamma*torch.linalg.norm(model.implicit.weight,2).item()
    print(f"Implicit: gamma={gamma} | width={width} | lr={lr} | opn={opn:.4f}")
    print(f"Fixed: norm_W={torch.linalg.norm(model.input.weight):.4f} | norm_B={torch.linalg.norm(model.output.weight):.4f}")

    # fix the first and output layer unchanged
    for name, param in model.named_parameters():
        if name == "implicit.input.weight":
            param.requires_grad = False

        if name == "output.weight":
            param.requires_grad = False

    for name, param in model.named_parameters():
        print(f"{name}, {param.requires_grad}")

    opt = optim.SGD(model.parameters(), lr=lr)
    
    train_errs, train_losses, opns = [], [], []
    test_errs, test_losses = [], []
    for i in range(max_iter):

        train_err, train_loss, opn = epoch(train_dataloader, model, opt, \
            lambda m: m.gamma* torch.linalg.norm(m.implicit.weight,2).item())
        test_err, test_loss, _ = epoch(test_dataloader, model)
        print(f"{i}: Forward: {model.iterations} | " + f"Train Error: {train_err:.4f}, Loss: {train_loss:.4f}, Operator norm: {opn:.4f} | " +
              f"Test Error: {test_err:.4f}, Loss: {test_loss:.4f}")
        train_errs.append(train_err), train_losses.append(train_loss), opns.append(opn)
        test_errs.append(test_err), test_losses.append(test_loss)

    grad_B = torch.norm(model.output.weight.grad,p='fro') if model.output.weight.grad != None else 0.0
    grad_A = torch.norm(model.implicit.implicit.weight.grad,p='fro')
    grad_W = torch.norm(model.implicit.input.weight.grad,p='fro') if model.implicit.input.weight.grad != None else 0.0

    # train_losses = [loss/train_losses[0] for loss in train_losses]
    # test_losses = [loss/test_losses[0] for loss in test_losses]

    return train_errs, train_losses, opns, test_errs, test_losses, grad_B, grad_A, grad_W

gamma = 0.5
width = 500
lr = 1/num

print("========start training=========")
test_list = [500]# [500, 1000, 1500, 2000]
train_errs, train_losses, opns = [], [], []
test_errs, test_losses = [], []
input_dim = 784
out_dim=10
model = ImplicitLayer(input_dim, width, out_dim,torch.relu,0.5,1.,max_iter = 10).to(device)

for x in test_list:
    print("===============================")
    torch.manual_seed(0)
    gamma = gamma
    width = x
    lr = lr
    train_err, train_losse, opn, test_err, test_losse, grad_B, grad_A, grad_W = train(model,gamma, input_dim, width, out_dim, train_data, test_data, lr)
    train_errs.append(train_err), train_losses.append(train_losse), opns.append(opn)
    test_errs.append(test_err), test_losses.append(test_losse)
    print(f"grad at the end: grad_B={grad_B:.4f}, grad_A={grad_A:.4f}, grad_W={grad_W:.4f}")

# classification error
plt.rc('font', size=20)  
plt.rc('axes', labelsize=20) 
plt.figure()
for i in range(len(test_list)):
    plt.plot(train_errs[i], label="width " + str(test_list[i]))

plt.title('train classification errors')
plt.ylabel('error')
plt.xlabel('epoch')
plt.legend(loc='upper right', prop={'size': 20})
plt.savefig('figs/MNIST/class_train_err.pdf', bbox_inches="tight")

plt.figure()
for i in range(len(test_list)):
    plt.plot(test_errs[i], label="width " + str(test_list[i]))

plt.rc('font', size=20)  
plt.rc('axes', labelsize=20) 
plt.title('test classification errors')
plt.ylabel('error')
plt.xlabel('epoch')
plt.legend(loc='upper right', prop={'size': 20})
plt.savefig('figs/MNIST/class_test_err.pdf', bbox_inches="tight")

# losses
plt.figure()
for i in range(len(test_list)):
    plt.plot(torch.log(torch.as_tensor(train_losses[i])), label="width " + str(test_list[i]))

plt.rc('font', size=20)  
plt.rc('axes', labelsize=20) 
plt.title('train loss')
plt.ylabel('log(loss)')
plt.xlabel('epoch')
plt.legend(loc='upper right', prop={'size': 20})
plt.savefig('figs/MNIST/train_model_loss.pdf', bbox_inches="tight")

plt.figure()
for i in range(len(test_list)):
    plt.plot(torch.log(torch.as_tensor(test_losses[i])), label="width " + str(test_list[i]))

plt.rc('font', size=20)  
plt.rc('axes', labelsize=20) 
plt.title('test loss')
plt.ylabel('log(loss)')
plt.xlabel('epoch')
plt.legend(loc='upper right', prop={'size': 20})
plt.savefig('figs/MNIST/test_model_loss.pdf', bbox_inches="tight")

# operator norms of implicit layer
plt.figure()
for i in range(len(test_list)):
    plt.plot(opns[i], label="width " + str(test_list[i]))

plt.rc('font', size=20)  
plt.rc('axes', labelsize=20) 
plt.title('operator norm of layer')
plt.ylabel('operator norm')
plt.xlabel('epoch')
plt.legend(loc='upper right', prop={'size': 20})
plt.savefig('figs/MNIST/opn.pdf', bbox_inches="tight")


# define the epoch
# a generic function for running a single epoch (training or evaluation)

# train_errs, train_losses, opns = [], [], []
# test_errs, test_losses = [], []
# for i in range(1000):

#     train_err, train_loss, opn = epoch(train_dataloader, model, opt, \
#         lambda m: m[2].gamma* torch.linalg.norm(m[2].linear.weight,2).item()/sqrt(m[2].out_features))
#     test_err, test_loss, _ = epoch(test_dataloader, model)
#     print(f"{i}: Forward: {model[2].iterations} | " + f"Train Error: {train_err:.4f}, Loss: {train_loss:.4f}, Operator norm: {opn:.4f} | " +
#           f"Test Error: {test_err:.4f}, Loss: {test_loss:.4f}")
#     train_errs.append(train_err), train_losses.append(train_loss), opns.append(opn)
#     test_errs.append(test_err), test_losses.append(test_loss)

# # classification errors
# plt.figure()
# plt.plot(train_errs, label="train")
# plt.plot(test_errs, label="test")
# plt.title('classification errors')
# plt.ylabel('error')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
# plt.savefig('figs/class_err.png')

# # cross entropy losses
# plt.figure()
# plt.plot(train_losses, label="train")
# plt.plot(test_losses, label="test")
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
# plt.savefig('figs/model_loss.png')

# # operator norms of implicit layer
# plt.figure()
# plt.plot(opns, label="operator norm")
# plt.title('operator norm of layer')
# plt.ylabel('operator norm')
# plt.xlabel('epoch')
# plt.savefig('figs/opn.png')


