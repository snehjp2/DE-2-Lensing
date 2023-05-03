import torch
import torch.nn as nn
from torch.nn import functional as F
from e2cnn import gspaces
from e2cnn import nn as e2cnn_nn
import torchvision

encoder_fields = [96, 48, 24]   

class Encoder(torch.nn.Module):
    def __init__(self, N, encoder_fields, reflections=False):
        super(Encoder, self).__init__()
        
        self.N = N
        self.encoder_fields = encoder_fields
        
        if (reflections == True) and (self.N == 1):
            self.r2_act = gspaces.Flip2dOnR2()

        elif reflections == True:
            self.r2_act = gspaces.FlipRot2dOnR2(N=self.N)

        else:
            self.r2_act = gspaces.Rot2dOnR2(N=self.N)
            
        in_type = e2cnn_nn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr]) ## 3 channels
        
        self.input_type = in_type
        
        out_type = e2cnn_nn.FieldType(self.r2_act, self.encoder_fields[0]*[self.r2_act.regular_repr])
        
        self.block1 = e2cnn_nn.SequentialModule(
            e2cnn_nn.MaskModule(in_type, 256, margin=1),
            e2cnn_nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False, stride=1),
            e2cnn_nn.InnerBatchNorm(out_type),
            e2cnn_nn.ReLU(out_type, inplace=True)
        )
        
        self.pool1 = e2cnn_nn.SequentialModule(
            e2cnn_nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        
        in_type = self.block1.out_type
        
        out_type = e2cnn_nn.FieldType(self.r2_act, self.encoder_fields[1]*[self.r2_act.regular_repr])
        
        self.block2 = e2cnn_nn.SequentialModule(
            e2cnn_nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False, stride=2),
            e2cnn_nn.InnerBatchNorm(out_type),
            e2cnn_nn.ReLU(out_type, inplace=True)
        )
        
        in_type = self.block2.out_type
        out_type = e2cnn_nn.FieldType(self.r2_act, self.encoder_fields[2]*[self.r2_act.regular_repr])

        
        self.block3 = e2cnn_nn.SequentialModule(
            e2cnn_nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False, stride=2),
            e2cnn_nn.InnerBatchNorm(out_type),
            e2cnn_nn.ReLU(out_type, inplace=True)
        )
        
        self.pool2 = e2cnn_nn.SequentialModule(
            e2cnn_nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        
        self.gpool = e2cnn_nn.GroupPooling(out_type)
        
    def forward(self, input: torch.Tensor):   
        x = e2cnn_nn.GeometricTensor(input, self.input_type)
        print(f'encoder input: {x.tensor.shape}')
        x = self.block1(x)
        print(f'encoder block1: {x.tensor.shape}')
        x = self.pool1(x)
        print(f'encoder pool1: {x.tensor.shape}')
        x = self.block2(x)
        print(f'encoder block2: {x.tensor.shape}')
        x = self.block3(x)
        print(f'encoder block3: {x.tensor.shape}')
        x = self.pool2(x)
        print(f'encoder pool3: {x.tensor.shape}')
        x = self.gpool(x)
        print(f'decoder input: {x.tensor.shape}')
        x = x.tensor
        
        return x 
    

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.ConvTranspose2d(24, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.conv_block2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.conv_block3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.conv_block4 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        x = self.conv_block1(x)
        print(f'decoder block1: {x.shape}')
        x = self.conv_block2(x)
        print(f'decoder block2: {x.shape}')
        x = self.conv_block3(x)
        print(f'decoder block3: {x.shape}')
        x = self.conv_block4(x)
        print(f'decoder block3: {x.shape}')
        return x
    
    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(N=2, encoder_fields=encoder_fields, reflections=True)
        self.decoder = Decoder()
    
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
    
    
def load_D4_encoder():
    D4_encoder = Encoder(N=4, encoder_fields=encoder_fields, reflections=True)
    return D4_encoder

def load_decoder():
    decoder = Decoder()
    return decoder
    
def load_autoencoder():
    autoencoder = Autoencoder()
    return autoencoder


model_dict = {'D4_encoder': load_D4_encoder,
              'decoder': load_decoder,
              'autoencoder': load_autoencoder}
    
if __name__ == '__main__':
        
    # encoder = Encoder(N=1, encoder_fields=encoder_fields, reflections=True)
    x = torch.rand(size=(1,3,256,256))
    autoencoder = Autoencoder()
    y = autoencoder(x)
   
    print(f'Trainable parameters = {sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)}')
    
        


