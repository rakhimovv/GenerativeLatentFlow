import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

import models.lr_scheduler as lr_scheduler
import models.networks as networks
from .base_model import BaseModel

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
        self.netF = networks.define_flow(opt).to(self.device)
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
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

            if train_opt['nll_weight'] is None:
                raise ValueError('nll loss should be always in this version')
            self.cri_nll = nn.L1Loss().to(self.device)  # F().to(device) # TODO FIX
            self.l_nll_w = train_opt['nll_weight']

            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None

            if self.cri_fea:
                self.netVGG = networks.define_VGG(opt, use_bn=False).to(self.device)
                if opt['dist']:
                    pass  # do not need to use DistributedDataParallel for netF
                else:
                    self.netVGG = DataParallel(self.netVGG)

            # optimizers
            self.optimizer_E = torch.optim.Adam(self.netE.parameters(),
                                                lr=train_opt['lr_E'],
                                                weight_decay=train_opt['weight_decay_E'] if train_opt[
                                                    'weight_decay_E'] else 0,
                                                betas=(train_opt['beta1_E'], train_opt['beta2_E']))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=train_opt['lr_D'],
                                                weight_decay=train_opt['weight_decay_D'] if train_opt[
                                                    'weight_decay_D'] else 0,
                                                betas=(train_opt['beta1_D'], train_opt['beta2_D']))
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(),
                                                lr=train_opt['lr_F'],
                                                weight_decay=train_opt['weight_decay_F'] if train_opt[
                                                    'weight_decay_F'] else 0,
                                                betas=(train_opt['beta1_F'], train_opt['beta2_F']))
            self.optimizers = [self.optimizer_E, self.optimizer_D, self.optimizer_F]

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
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        self.print_network()  # print networks structure
        self.load()  # load G, D, F if needed

    def feed_data(self, data, need_GT=True):
        self.image = data['image'].to(self.device)
        if need_GT:
            self.image_gt = self.image

    def optimize_parameters(self, step):
        self.optimizer_E.zero_grad()
        self.optimizer_D.zero_grad()
        self.optimizer_F.zero_grad()

        z = self.encoder(self.image)
        reconstructed = self.decoder(z)

        l_total = 0

        if self.cri_pix:  # pixel loss
            l_pix = self.l_pix_w * self.cri_pix(self.reconstructed, self.image_gt)
            l_total += l_pix

        if self.cri_fea:  # feature loss
            gt_fea = self.netVGG(self.image_gt).detach()
            reconstructed_fea = self.netVGG(reconstructed)
            l_fea = self.l_fea_w * self.cri_fea(reconstructed_fea, gt_fea)
            l_total += l_fea

        # negative likelihood loss
        noise_out = self.netF(z.detach())
        l_nll = self.l_fea_w * self.cri_nll(noise_out)
        l_total += l_nll

        l_total.backward()
        self.optimizer_E.step()
        self.optimizer_D.step()
        self.optimizer_F.step()

        # set log
        if self.cri_pix:
            self.log_dict['l_pix'] = l_pix.item()
        if self.cri_fea:
            self.log_dict['l_fea'] = l_fea.item()
        if self.cri_nll:
            self.log_dict['l_nll'] = l_nll.item()

    def test(self):
        self.netE.eval()
        self.netD.eval()
        self.netF.eval()

        with torch.no_grad():
            noise = torch.randn()
            self.sampled_images = self.netF.reverse(noise)
            self.reconstructed = self.netD(self.netE(self.image))

        # TODO write logging

        self.netE.train()
        self.netD.train()
        self.netF.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['sampled'] = self.sampled_images.detach()[0].float().cpu()
        return out_dict

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

        if self.is_train:
            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netVGG)
                if isinstance(self.netVGG, nn.DataParallel) or isinstance(
                        self.netVGG, DistributedDataParallel):
                    net_struc_str = '{} - {}'.format(self.netVGG.__class__.__name__,
                                                     self.netVGG.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netVGG.__class__.__name__)
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
