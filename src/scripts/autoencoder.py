import torch
import torch.nn as nn
from torch.nn import functional as F
from escnn import gspaces
from escnn import nn as escnn_nn
import torchvision
import lightning.pytorch as pl
import argparse
import yaml
# encoder_fields = [96, 48, 24]


class EquivariantEncoder(torch.nn.Module):
    def __init__(self, N, encoder_fields=[48, 24, 16], reflections=False):
        super(EquivariantEncoder, self).__init__()

        self.N = N
        self.encoder_fields = encoder_fields

        if (reflections == True) and (self.N == 1):
            self.r2_act = gspaces.flip2dOnR2()

        elif reflections == True:
            self.r2_act = gspaces.flipRot2dOnR2(N=self.N)

        else:
            self.r2_act = gspaces.rot2dOnR2(N=self.N)

        in_type = escnn_nn.FieldType(
            self.r2_act, 3*[self.r2_act.trivial_repr])  # 3 channels

        self.input_type = in_type

        out_type = escnn_nn.FieldType(
            self.r2_act, self.encoder_fields[0]*[self.r2_act.regular_repr])

        self.block1 = escnn_nn.SequentialModule(
            escnn_nn.MaskModule(in_type, 256, margin=1),
            escnn_nn.R2Conv(in_type, out_type, kernel_size=5,
                            padding=2, bias=False, stride=1),
            escnn_nn.InnerBatchNorm(out_type),
            escnn_nn.ReLU(out_type, inplace=True)
        )

        self.pool1 = escnn_nn.SequentialModule(
            escnn_nn.PointwiseAvgPoolAntialiased(
                out_type, sigma=0.66, stride=2)
        )

        in_type = self.block1.out_type

        out_type = escnn_nn.FieldType(
            self.r2_act, self.encoder_fields[1]*[self.r2_act.regular_repr])

        self.block2 = escnn_nn.SequentialModule(
            escnn_nn.R2Conv(in_type, out_type, kernel_size=5,
                            padding=2, bias=False, stride=2),
            escnn_nn.InnerBatchNorm(out_type),
            escnn_nn.ReLU(out_type, inplace=True)
        )

        in_type = self.block2.out_type
        out_type = escnn_nn.FieldType(
            self.r2_act, self.encoder_fields[2]*[self.r2_act.regular_repr])

        self.block3 = escnn_nn.SequentialModule(
            escnn_nn.R2Conv(in_type, out_type, kernel_size=5,
                            padding=2, bias=False, stride=2),
            escnn_nn.InnerBatchNorm(out_type),
            escnn_nn.ReLU(out_type, inplace=True)
        )

        self.pool2 = escnn_nn.SequentialModule(
            escnn_nn.PointwiseAvgPoolAntialiased(
                out_type, sigma=0.66, stride=2)
        )

        self.gpool = escnn_nn.GroupPooling(out_type)

    def forward(self, input: torch.Tensor):
        x = escnn_nn.GeometricTensor(input, self.input_type)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool2(x)
        x = self.gpool(x)
        x = x.tensor

        return x


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv_block2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.conv_block3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.conv_block4 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        return x


class LinearDecoder(nn.Module):
    def __init__(self):
        super(LinearDecoder, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(nn.Linear(32*32*64, 512),
                                nn.ReLU(inplace=True),
                                nn.Linear(512, 128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 32),
                                nn.ReLU(inplace=True),
                                nn.Linear(32, 32),
                                nn.ReLU(inplace=True),
                                nn.Linear(32, 5))

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Autoencoder(pl.LightningModule):

    def __init__(self, config):
        super(Autoencoder, self).__init__()
        self.encoder = EquivariantEncoder(N=4, reflections=True)
        self.decoder1 = Decoder()
        self.decoder2 = LinearDecoder()

        self.config = config
        self.val_img_losses = []
        self.val_param_losses = []
        self.val_losses = []

    def forward(self, x):
        x = self.encoder(x)
        x1 = self.decoder1(x)
        x2 = self.decoder2(x)
        return x1, x2

    def training_step(self, batch, batch_idx):
        original_image, lensed_image, label, params = batch
        pred_img, pred_params = self(lensed_image)
        img_loss = F.mse_loss(pred_img, original_image)
        param_loss = F.mse_loss(pred_params, params)
        loss = img_loss + param_loss
        self.log('train_img_loss', img_loss,on_step=True, on_epoch=True, logger=True)
        self.log('train_param_loss', param_loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        original_image, lensed_image, label, params = batch
        pred_img, pred_params = self(lensed_image)
        img_loss = F.mse_loss(pred_img, original_image)
        param_loss = F.mse_loss(pred_params, params)
        loss = img_loss + param_loss

        self.val_img_losses.append(img_loss)
        self.val_param_losses.append(param_loss)
        self.val_losses.append(loss)

    
    def on_validation_epoch_end(self):
        self.log('val_img_loss', torch.stack(self.val_img_losses).mean(), on_step=False, on_epoch=True, logger=True)
        self.log('val_param_loss', torch.stack(self.val_param_losses).mean(), on_step=False, on_epoch=True, logger=True)
        self.log('val_loss', torch.stack(self.val_losses).mean(), on_step=False, on_epoch=True, logger=True)

        self.val_img_losses.clear()
        self.val_param_losses.clear()
        self.val_losses.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config['parameters']['lr'],
                                      weight_decay=self.config['parameters']['weight_decay'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.config['parameters']['milestones'],
                                                            gamma=self.config['parameters']['lr_decay'])
        return [optimizer], [scheduler]




def load_autoencoder(config):
    autoencoder = Autoencoder(config)
    return autoencoder


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'Train the models')
    parser.add_argument('--config', metavar = 'config', required=True,
                    help='Location of the config file')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    autoencoder = load_autoencoder(config)

    x = torch.rand(size=(1, 3, 256, 256))
    img, vec = autoencoder(x)
    print(img.shape)
    print(vec.shape)
    print(f'Trainable parameters = {sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)}')
