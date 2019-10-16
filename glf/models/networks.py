import torch

import models.archs.GLF_arch as GLF_arch
import models.archs.vgg_arch as discriminator_vgg_arch


# Encoder
def define_encoder(opt):
    opt_net = opt['encoder']
    which_model = opt_net['which_model']

    if which_model == 'glf_encoder':
        netE = GLF_arch.Encoder(img_size=opt_net['img_size'],
                                in_ch=opt_net['in_ch'],
                                nz=opt_net['nz'])
    # elif which_model == 'vae_encoder':
    else:
        raise NotImplementedError('Encoder model [{:s}] not recognized'.format(which_model))

    return netE


# Decoder
def define_decoder(opt):
    opt_net = opt['decoder']
    which_model = opt_net['which_model']

    if which_model == 'glf_decoder':
        netD = GLF_arch.Decoder(img_size=opt_net['img_size'],
                                out_ch=opt_net['out_ch'],
                                nz=opt_net['nz'])
    # elif which_model == 'vae_decoder':
    else:
        raise NotImplementedError('Decoder model [{:s}] not recognized'.format(which_model))

    return netD


# Flow
def define_flow(opt):
    opt_net = opt['flow']
    which_model = opt_net['which_model']

    if which_model == 'FlowNet':
        netD = GLF_arch.FlowNet(nz=opt_net['nz'],
                                hidden_size=opt_net['hidden_size'],
                                nblocks=opt_net['nblocks'])
    # elif which_model == 'hamiltonian_flow':
    else:
        raise NotImplementedError('Flow model [{:s}] not recognized'.format(which_model))

    return netD


# Feature extractor used for perceptual loss
def define_VGG(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netVGG = discriminator_vgg_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                                        use_input_norm=True, device=device)
    netVGG.eval()  # No need to train
    return netVGG
