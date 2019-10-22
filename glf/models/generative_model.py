import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

import glf.models.lr_scheduler as lr_scheduler
import glf.models.networks as networks
from glf.models.archs.vgg_arch import VGGLoss
from glf.models.base_model import BaseModel
from glf.models.loss import NLLLoss

logger = logging.getLogger('base')


class GenerativeModel(BaseModel):
    def __init__(self, opt):
        super(GenerativeModel, self).__init__(opt)

        # DISTRIBUTED TRAINING OR NOT
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1

        # DEFINE NETWORKS
        self.netE = networks.define_encoder(opt).to(self.device)
        self.netD = networks.define_decoder(opt).to(self.device)
        self.netF, self.nz, self.stop_gradients = networks.define_flow(opt)
        self.netF.to(self.device)
        if opt['dist']:
            self.netE = DistributedDataParallel(self.netE, device_ids=[torch.cuda.current_device()])
            self.netD = DistributedDataParallel(self.netD, device_ids=[torch.cuda.current_device()])
            self.netF = DistributedDataParallel(self.netF, device_ids=[torch.cuda.current_device()])
        else:
            self.netE = DataParallel(self.netE)
            self.netD = DataParallel(self.netD)
            self.netF = DataParallel(self.netF)

        if self.is_train:
            self.netE.train()
            self.netD.train()
            self.netF.train()

        # GET CONFIG PARAMS FOR LOSSES AND LR
        train_opt = opt['train']

        # DEFINE LOSSES, OPTIMIZER AND SCHEDULE
        if self.is_train:
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss(reduction='mean').to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss(reduction='mean').to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']

                if train_opt['add_background_mask']:
                    self.add_mask = True
                else:
                    self.add_mask = False

            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

            if train_opt['nll_weight'] is None:
                raise ValueError('nll loss should be always in this version')
            self.cri_nll = NLLLoss(reduction='mean').to(self.device)  # F().to(device) # TODO FIX
            self.l_nll_w = train_opt['nll_weight']

            if train_opt['feature_weight'] > 0:
                self.cri_fea = VGGLoss().to(self.device)
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None

            # optimizers
            if train_opt['lr_E'] > 0:
                self.optimizer_E = torch.optim.Adam(self.netE.parameters(),
                                                    lr=train_opt['lr_E'],
                                                    weight_decay=train_opt['weight_decay_E'] if train_opt[
                                                        'weight_decay_E'] else 0,
                                                    betas=(train_opt['beta1_E'], train_opt['beta2_E']))
                self.optimizers.append(self.optimizer_E)
            else:
                for p in self.netE.parameters():
                    p.requires_grad_(False)
                logger.info('Freeze encoder.')

            if train_opt['lr_D'] > 0:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                    lr=train_opt['lr_D'],
                                                    weight_decay=train_opt['weight_decay_D'] if train_opt[
                                                        'weight_decay_D'] else 0,
                                                    betas=(train_opt['beta1_D'], train_opt['beta2_D']))
                self.optimizers.append(self.optimizer_D)
            else:
                for p in self.netD.parameters():
                    p.requires_grad_(False)
                logger.info('Freeze decoder.')

            if train_opt['lr_F'] > 0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(),
                                                    lr=train_opt['lr_F'],
                                                    weight_decay=train_opt['weight_decay_F'] if train_opt[
                                                        'weight_decay_F'] else 0,
                                                    betas=(train_opt['beta1_F'], train_opt['beta2_F']))
                self.optimizers.append(self.optimizer_F)
            else:
                for p in self.netF.parameters():
                    p.requires_grad_(False)
                logger.info('Freeze flow.')

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                logger.info('No learning rate scheme is applied.')

            self.log_dict = OrderedDict()

        self.print_network()  # print networks structure
        self.load()  # load G, D, F if needed

        # CHECK THAT FLOW WORKS CORRECTLY
        with torch.no_grad():
            test_input = torch.randn((2, self.nz)).to(self.device)
            test_output, test_logdet = self.netF(test_input)
            if isinstance(self.netF, nn.DataParallel) or isinstance(self.netF, DistributedDataParallel):
                test_input2 = self.netF.module.reverse(test_output)
            else:
                test_input2 = self.netF.reverse(test_output)
            assert torch.allclose(test_input, test_input2), 'Flow model is incorrect'
            del test_input, test_output, test_input2, test_logdet

    def feed_data(self, data, need_GT=True):
        self.image = data[0].to(self.device)
        if need_GT:
            self.image_gt = self.image

    def optimize_parameters(self, step):

        for optimizer in self.optimizers:
            optimizer.zero_grad()

        z = self.netE(self.image)
        reconstructed = self.netD(z)

        l_total = 0

        if self.cri_pix:  # pixel loss
            if self.add_mask:
                mask = (self.image_gt[:, 0, :, :] == 1).unsqueeze(1).float()
                inv_mask = 1 - mask
                l_pix = (1 * self.cri_pix(reconstructed * mask, self.image_gt * mask) +
                         10 * self.cri_pix(reconstructed * inv_mask, self.image_gt * inv_mask))
            else:
                l_pix = self.l_pix_w * self.cri_pix(reconstructed, self.image_gt)
            l_total += l_pix

        if self.cri_fea:  # feature loss
            l_fea = self.l_fea_w * self.cri_fea(reconstructed, self.image_gt)
            l_total += l_fea

        # negative likelihood loss
        if self.stop_gradients:
            noise_out, logdets = self.netF(z.detach())
        else:
            noise_out, logdets = self.netF(z)

        l_nll = self.l_nll_w * self.cri_nll(noise_out, logdets)
        l_total += l_nll

        l_total.backward()
        for optimizer in self.optimizers:
            optimizer.step()

        # set log
        if self.cri_pix:
            self.log_dict['l_pix'] = l_pix.item()
        if self.cri_fea:
            self.log_dict['l_fea'] = l_fea.item()
        if self.cri_nll:
            self.log_dict['l_nll'] = l_nll.item()

    def sample_images(self, n=25):
        self.netF.eval()
        self.netD.eval()
        with torch.no_grad():
            noise = torch.randn(n, self.nz).to(self.device)
            if isinstance(self.netF, nn.DataParallel) or isinstance(self.netF, DistributedDataParallel):
                sample = self.netD(self.netF.module.reverse(noise)).detach().float().cpu()
            else:
                sample = self.netD(self.netF.reverse(noise)).detach().float().cpu()
        self.netF.train()
        self.netD.train()
        return sample

    # def test(self):
    #     self.netE.eval()
    #     self.netD.eval()
    #     self.netF.eval()
    #
    #     self.netE.train()
    #     self.netD.train()
    #     self.netF.train()

    def get_current_log(self):
        return self.log_dict

    def print_network(self):
        for name, net in [('E', self.netE), ('D', self.netD), ('F', self.netF)]:
            s, n = self.get_network_description(net)
            if isinstance(net, nn.DataParallel) or isinstance(net, DistributedDataParallel):
                net_struc_str = '{} - {}'.format(net.__class__.__name__,
                                                 net.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(net.__class__.__name__)
            if self.rank <= 0:
                logger.info('Network {} structure: {}, with parameters: {:,d}'.format(name, net_struc_str, n))
                logger.info(s)

        if self.is_train and self.cri_fea:
            vgg_net = self.cri_fea.vgg
            s, n = self.get_network_description(vgg_net)
            if isinstance(vgg_net, nn.DataParallel) or isinstance(vgg_net, DistributedDataParallel):
                net_struc_str = '{} - {}'.format(vgg_net.__class__.__name__,
                                                 vgg_net.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(vgg_net.__class__.__name__)
            if self.rank <= 0:
                logger.info('Network VGG structure: {}, with parameters: {:,d}'.format(
                    net_struc_str, n))
                logger.info(s)

    def load(self):
        load_path_E = self.opt['path']['pretrained_encoder']
        if load_path_E is not None:
            logger.info('Loading model for E [{:s}] ...'.format(load_path_E))
            self.load_network(load_path_E, self.netE, self.opt['path']['strict_load'])

        load_path_D = self.opt['path']['pretrained_decoder']
        if load_path_D is not None:
            logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD, self.opt['path']['strict_load'])

        load_path_F = self.opt['path']['pretrained_flow']
        if load_path_F is not None:
            logger.info('Loading model for F [{:s}] ...'.format(load_path_F))
            self.load_network(load_path_F, self.netF, self.opt['path']['strict_load'])

    def save(self, iter_step):
        self.save_network(self.netE, 'G', iter_step)
        self.save_network(self.netD, 'D', iter_step)
        self.save_network(self.netF, 'F', iter_step)
