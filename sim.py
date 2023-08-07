import torch
import torch.nn as nn
import torch.nn.functional as F # for relu
import torch.linalg
import torch.nn.init as init
from math import sqrt
from torch.utils.data import DataLoader
from load_dataset import load_mnist
import torch.optim as optim
import pandas as pd
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import argparse
import tensorflow.compat.v1 as tf
## Define the implicit model 
## params:
##    sigma_w: variance of W, default 0.6
##    sigma_u: variance of initial matrix U, default 1.0
##    max_iter: depth, default 50
##    act: activation function, default tanh
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

        init.normal_(self.output.weight, mean=0, std=1./sqrt(input_features)) # initialization of W
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

        yp = model(X)
        # loss = nn.CrossEntropyLoss()(yp,y)
        # one_hot = torch.nn.functional.one_hot(y, yp.shape[1]) #already onehot no need
        loss = criterion(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_err += (yp.max(dim=1)[1] != y.max(dim=1)[1]).sum().item() 
        #total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
        if monitor is not None:
            total_monitor += monitor(model)
    return total_err / len(loader.dataset), total_loss / len(loader.dataset), total_monitor / len(loader) 


# need to create a method to take width as dataset dim, implicit layers, max iteration, output dim
def train(model,input_dim, width, out_dim, train_dataloader, test_dataloader, lr, max_iter=10000):
    # model = nn.Sequential(nn.Flatten(),
    #     nn.Linear(input_dim, width, bias=False),
    #     reluImplicitLayer(width, gamma = gamma, max_iter=100),
    #     nn.Linear(width, out_dim, bias=False)).to(device)

    opn = torch.linalg.norm(model.implicit.weight,2).item()  #model.gamma*torch.linalg.norm(model.implicit.implicit.weight,2).item()
    #print(f"Implicit: gamma={gamma} | width={width} | lr={lr} | opn={opn:.4f}")
    print(f"width={width} | lr={lr} | opn={opn:.4f}")
    print(f"Fixed: norm_W={torch.linalg.norm(model.input.weight):.4f} | norm_B={torch.linalg.norm(model.output.weight):.4f}")

    # fix the first and output layer unchanged
    for name, param in model.named_parameters():
        if name == "input.weight":
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
            lambda m:  torch.linalg.norm(m.implicit.weight,2).item())
        test_err, test_loss, _ = epoch(test_dataloader, model)
        #print(f"{i}: Forward: {model.iterations} | " + f"Train Error: {train_err:.4f}, Loss: {train_loss:.4f}, Operator norm: {opn:.4f} | " +
        #      f"Test Error: {test_err:.4f}, Loss: {test_loss:.4f}")
        #train_errs.append(train_err), train_losses.append(train_loss), opns.append(opn)
        #test_errs.append(test_err), test_losses.append(test_loss)

    grad_B = torch.norm(model.output.weight.grad,p='fro') if model.output.weight.grad != None else 0.0
    grad_A = torch.norm(model.implicit.weight.grad,p='fro').item()
    grad_W = torch.norm(model.input.weight.grad,p='fro') if model.input.weight.grad != None else 0.0

    # train_losses = [loss/train_losses[0] for loss in train_losses]
    # test_losses = [loss/test_losses[0] for loss in test_losses]

    return train_err, train_loss, opn, test_err, test_loss, grad_B, grad_A, grad_W

#for constructing dataloader
class training_set(torch.utils.data.Dataset):
    def __init__(self,X,Y):
        self.X = X                           # set data
        self.Y = Y                           # set lables

    def __len__(self):
        return len(self.X)                   # return length

    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]]    # return list of batch data [data, labels]

for _ in range(1):
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('num_train', 1000, 'Number of training data.')
    flags.DEFINE_integer('batch_size',100,"ba")
    flags.DEFINE_float('sigma_w',0.6,"sigma_w")
    flags.DEFINE_float('sigma_u',1.0,"sigma_u")
    flags.DEFINE_integer('width',1000,"width")
    flags.DEFINE_integer('depth',30,"depth")
    flags.DEFINE_float('lr',0.01,"lr")
    flags.DEFINE_integer('epochs',10000,"epochs")
    flags.DEFINE_string('act','tanh',"act")
    #parser = argparse.ArgumentParser()
def main(args):
    #parser.add_argument('--batch_size',type=int,default=100)
    #parser.add_argument('--sigma_w',type=float,default=0.6)
    #parser.add_argument('--sigma_u',type=float,default=1.0)
    #parser.add_argument('--act',type=str,default='relu')
    #parser.add_argument('--width',type=int,default=1000)
    #parser.add_argument('--depth',type=int,default=30)
    #parser.add_argument('--lr',type=int,default=0.01)
    #parser.add_argument('--num_train',type=int,default=100)
    #parser.add_argument('--epochs',type=int,default=10000)
    #args = parser.parse_args()
    
    batch_size = FLAGS.batch_size
    num_train = FLAGS.num_train
    sigma_w = FLAGS.sigma_w
    sigma_u = FLAGS.sigma_u
    width = FLAGS.width
    depth = FLAGS.depth
    lr = FLAGS.lr
    epochs = FLAGS.epochs
    if FLAGS.act=='relu':
        act = torch.relu
    elif FLAGS.act=='tanh':
        act = torch.tanh
    else:
        raise NotImplementedError
    print(width)

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
    sigma_w = 0.6
    input_dim = train_image.shape[-1]
    output_dim = train_label.shape[-1]
    model = ImplicitLayer(input_dim, width, output_dim,act,sigma_w,sigma_u,max_iter = depth)
    model.to(device)
    torch.manual_seed(0)
    print(model(train_image.to(device)).shape)
    print(train_label.shape)
    
    train_err, train_losses, opn, test_err, test_losses, grad_B, grad_A, grad_W = train(model, input_dim, width, output_dim, train_dataloader, test_dataloader, lr, max_iter=epochs)
    print(f"grad at the end: grad_B={grad_B:.4f}, grad_A={grad_A:.4f}, grad_W={grad_W:.4f}")
    df = pd.DataFrame({'num_train':num_train,"width":width,"depth":depth,"sigma_w":sigma_w,"sigma_u":sigma_u,  'train_err': train_err, 'train_losses': train_losses, "opn":opn, "test_err":test_err, "test_losses":test_losses, "gradA":grad_A, "gradW":grad_W},index=[0])
    df.to_csv('tmp/sim/results.csv', mode='a', header=False)


    
if __name__=='__main__':
    tf.app.run(main)
