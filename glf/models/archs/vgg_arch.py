import torch
import torch.nn as nn
import torchvision
from torch.nn.parallel import DataParallel


class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_ids=(2, 7, 12, 21, 30), use_bn=False, use_input_norm=True, device=torch.device('cpu'),
                 vgg_v=16):
        super(VGGFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        if vgg_v == 19 and use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        elif vgg_v == 19:
            model = torchvision.models.vgg19(pretrained=True)
        elif vgg_v == 16 and use_bn:
            model = torchvision.models.vgg16_bn(pretrained=True)
        elif vgg_v == 16:
            model = torchvision.models.vgg16(pretrained=True)
        else:
            raise ValueError

        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485 - 1, 0.456 - 1, 0.406 - 1] if input in range [-1, 1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229 * 2, 0.224 * 2, 0.225 * 2] if input in range [-1, 1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        self.feature_layers_num = len(layer_ids)
        prev_feature_layer = 0
        for i, feature_layer in enumerate(layer_ids):
            m = nn.Sequential(*list(model.features.children())[prev_feature_layer:feature_layer + 1])
            # No need to BP to variable
            for k, v in m.named_parameters():
                v.requires_grad = False
            setattr(self, 'fe{}'.format(i), m)
            prev_feature_layer = feature_layer + 1

    def forward(self, x):
        # Input range must be in [0, 1] or [-1, 1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        output = []
        for i in range(self.feature_layers_num):
            fe = getattr(self, 'fe{}'.format(i))
            if i == 0:
                output.append(fe(x))
            else:
                output.append(fe(output[i - 1]))
        return output


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_ids = (2, 7, 12, 21, 30)
        self.vgg = DataParallel(VGGFeatureExtractor(layer_ids=self.layer_ids))
        self.vgg.eval()
        self.l1_loss = nn.L1Loss(reduction='mean')

    def forward(self, input, target):
        features_input = self.vgg(input)
        features_target = self.vgg(target)
        perceptual_loss = 0
        for i in range(len(self.layer_ids)):
            perceptual_loss += self.l1_loss(features_input[i], features_target[i]) / len(self.layer_ids)
        return perceptual_loss


def test():
    img1 = torch.rand(256, 3, 32, 32)
    img2 = torch.rand(256, 3, 32, 32)
    vgg_loss = VGGLoss()
    result = vgg_loss(img1, img2)
    print(result, result.shape)
