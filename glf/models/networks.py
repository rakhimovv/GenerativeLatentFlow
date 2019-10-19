import glf.models.archs.GLF_arch as GLF_arch


# Encoder
def define_encoder(opt):
    opt_net = opt['encoder']
    which_model = opt_net['which_model']

    if which_model == 'glf_encoder':
        netE = GLF_arch.Encoder(img_size=opt_net['img_size'],
                                in_ch=opt_net['in_ch'],
                                nz=opt_net['nz'])
    # elif which_model == 'vae_encoder':
    elif which_model == 'glf_tiny_encoder':
        netE = GLF_arch.TinyEncoder(img_size=opt_net['img_size'],
                                in_ch=opt_net['in_ch'],
                                nz=opt_net['nz'])
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
    elif which_model == 'glf_tiny_decoder':
        netD = GLF_arch.TinyDecoder(img_size=opt_net['img_size'],
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

    return netD, opt_net['nz']
