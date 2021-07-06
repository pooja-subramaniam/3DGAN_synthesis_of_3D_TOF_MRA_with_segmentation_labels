import torch.nn as nn
import config as c


# custom weights initialization called on netG and netD
# based on Neff et al. 2017 parameters
def weights_init(model):
    classname = model.__class__.__name__

    if classname.find('Conv') != -1:
        mean = 0.0
        std = 0.05
        nn.init.normal_(model.weight.data, mean, std)
    elif classname.find('BatchNorm') != -1:
        mean = 0.0
        std = 0.05
        nn.init.normal_(model.weight.data, mean, std)
        nn.init.constant_(model.bias.data, 0)


# Generator block
class Generator(nn.Module):

    def __init__(self, activation='relu'):
        super(Generator, self).__init__()

        self.init_size = c.image_size[0] // 8
        self.init_z = c.image_size[-1] // 8

        activations = nn.ModuleDict([['lrelu', nn.LeakyReLU(0.2,
                                                            inplace=True)],
                                     ['relu', nn.ReLU(inplace=True)]])

        def upsample_conv_block(in_channels, out_channels,
                                activation=activation):
            if not c.spectral_norm_G:
                block = [nn.Upsample(scale_factor=2),
                         nn.Conv3d(in_channels, out_channels, c.kg, stride=1,
                                   padding=(c.kg-1)//2),
                         nn.BatchNorm3d(out_channels)]
            else:
                block = [nn.Upsample(scale_factor=2),
                         nn.utils.spectral_norm(nn.Conv3d(in_channels,
                                                          out_channels,
                                                          c.kg, stride=1,
                                                          padding=(c.kg-1)//2))
                         ]

            block.append(activations[activation])

            return block

        self.linear1 = nn.Sequential(nn.Linear(c.nz, c.ngf *
                                               (self.init_size ** 2) *
                                               self.init_z))
        self.batch1 = nn.Sequential(nn.BatchNorm3d(c.ngf),
                                    activations[activation])

        self.layer2 = nn.Sequential(*upsample_conv_block(c.ngf, c.ngf//2))
        self.layer3 = nn.Sequential(*upsample_conv_block(c.ngf//2, c.ngf//4))
        self.layer4 = nn.Sequential(*upsample_conv_block(c.ngf//4, c.ngf//8))
        self.layer5 = nn.Conv3d(c.ngf // 8, c.nc, c.kg, stride=1,
                                padding=(c.kg-1)//2)
        self.activationG = nn.Tanh()

    def forward(self, inp):
        # print(inp.size())
        x = self.linear1(inp.view(inp.size()[0], -1))
        x = x.view(x.size()[0], c.ngf, self.init_size, self.init_size,
                   self.init_z)
        x = self.batch1(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        gen_image = self.activationG(x)
        return gen_image


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        def conv_block(in_channels, out_channels, activation='LeakyReLU'):
            if not c.spectral_norm_D:
                block = [nn.Conv3d(in_channels, out_channels, c.kd, stride=2,
                                   padding=(c.kd-1)//2),
                         nn.InstanceNorm3d(out_channels)]
            else:
                block = [nn.utils.spectral_norm(nn.Conv3d(in_channels,
                                                          out_channels,
                                                          c.kd, stride=2,
                                                          padding=(c.kd-1)//2))
                         ]

            if activation == 'LeakyReLU':
                block.append(nn.LeakyReLU(0.2, inplace=True))
            else:
                block.append(nn.ReLU(inplace=True))
            return block

        self.layer1 = nn.Sequential(*conv_block(c.nc, c.ndf//8))
        self.layer2 = nn.Sequential(*conv_block(c.ndf//8, c.ndf//4))
        self.layer3 = nn.Sequential(*conv_block(c.ndf//4, c.ndf//2))
        self.layer4 = nn.Sequential(*conv_block(c.ndf//2, c.ndf))
        self.layer5 = nn.Linear(c.ndf * 8 * 8 * 4, 1)

    def forward(self, inp):
        x = self.layer1(inp)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        disc_out = self.layer5(x.view(x.size()[0], -1))
        return disc_out
