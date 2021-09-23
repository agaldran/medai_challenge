import sys
import segmentation_models_pytorch as smp
import torch

class SMP_W(torch.nn.Module):
    def __init__(self, decoder = smp.FPN, encoder_name='resnet34', encoder2_name=None, in_channels=3,
                                          encoder_weights='imagenet', classes=1, mode='train'):
        super(SMP_W, self).__init__()
        if encoder2_name is None: encoder2_name=encoder_name
        self.m1 = decoder(encoder_name=encoder_name, in_channels=in_channels, encoder_weights=encoder_weights, classes=classes)
        self.m2 = decoder(encoder_name=encoder2_name, in_channels=in_channels+classes, encoder_weights=encoder_weights, classes=classes)
        self.n_classes = classes
        self.mode=mode
    def forward(self, x):
        x1 = self.m1(x)
        x2 = self.m2(torch.cat([x, x1], dim=1))
        if self.mode!='train':
            return x2
        return x1,x2

def get_arch(model_name, in_c=3, n_classes=1, pretrained=True):

    e_ws = 'imagenet' if pretrained else None

    ## FPNET ##
    if model_name == 'fpnet_resnet18':
        model = smp.FPN(encoder_name='resnet18', in_channels=in_c, classes=n_classes)
    elif model_name == 'fpnet_resnet18_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='resnet18', in_channels=in_c, classes=n_classes, encoder_weights=e_ws)

    elif model_name == 'fpnet_mobilenet':
        model = smp.FPN(encoder_name='mobilenet_v2', in_channels=in_c, classes=n_classes)
    elif model_name == 'fpnet_mobilenet_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='mobilenet_v2', in_channels=in_c, classes=n_classes, encoder_weights=e_ws)

    elif model_name == 'fpnet_resnet34':
        model = smp.FPN(encoder_name='resnet34', in_channels=in_c, classes=n_classes)
    elif model_name == 'fpnet_resnet34_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='resnet34', in_channels=in_c, classes=n_classes, encoder_weights=e_ws)

    elif model_name == 'fpnet_resnet50':
        model = smp.FPN(encoder_name='resnet50', in_channels=in_c, classes=n_classes)
    elif model_name == 'fpnet_resnet50_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='resnet50', in_channels=in_c, classes=n_classes, encoder_weights=e_ws)

    elif model_name == 'fpnet_dpn68':
        model = smp.FPN(encoder_name='dpn68', in_channels=in_c, classes=n_classes)
    elif model_name == 'fpnet_dpn68_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='dpn68', in_channels=in_c, classes=n_classes, encoder_weights=e_ws)

    elif model_name == 'fpnet_dpn92':
        model = smp.FPN(encoder_name='dpn92', in_channels=in_c, classes=n_classes, encoder_weights='imagenet+5k')
    elif model_name == 'fpnet_dpn92_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='dpn92', in_channels=in_c, classes=n_classes, encoder_weights='imagenet+5k')
        
    elif model_name == 'fpnet_densenet121':
        model = smp.FPN(encoder_name='densenet121', in_channels=in_c, classes=n_classes)
    elif model_name == 'fpnet_densenet121_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='densenet121', in_channels=in_c, classes=n_classes)

    elif model_name == 'fpnet_densenet169':
        model = smp.FPN(encoder_name='densenet169', in_channels=in_c, classes=n_classes)
    elif model_name == 'fpnet_densenet169_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='densenet169', in_channels=in_c, classes=n_classes)
        
    # ADDITIONS FOR THIS PROJECT
    elif model_name == 'fpnet_resnext50':
        model = smp.FPN(encoder_name='resnext50_32x4d', in_channels=in_c, classes=n_classes)
    elif model_name == 'fpnet_resnext50_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='resnext50_32x4d', in_channels=in_c, classes=n_classes, encoder_weights=e_ws)

    elif model_name == 'UnetPlusPlus_dpn68_W':
        model = SMP_W(decoder=smp.UnetPlusPlus, encoder_name='dpn68', in_channels=in_c, classes=n_classes, encoder_weights=e_ws)
    elif model_name == 'UnetPlusPlus_dpn92_W':
        model = SMP_W(decoder=smp.UnetPlusPlus, encoder_name='dpn92', in_channels=in_c, classes=n_classes, encoder_weights='imagenet+5k')
    elif model_name == 'UnetPlusPlus_resnet34_W':
        model = SMP_W(decoder=smp.UnetPlusPlus, encoder_name='resnet34', in_channels=in_c, classes=n_classes, encoder_weights=e_ws)
    elif model_name == 'UnetPlusPlus_resnext50_W':
        model = SMP_W(decoder=smp.UnetPlusPlus, encoder_name='resnext50_32x4d', in_channels=in_c, classes=n_classes, encoder_weights=e_ws)
    elif model_name == 'UnetPlusPlus_mobilenet_W':
        model = SMP_W(decoder=smp.UnetPlusPlus, encoder_name='mobilenet_v2', in_channels=in_c, classes=n_classes, encoder_weights=e_ws)

    elif model_name == 'PSPNet_mobilenetv3_large_W':
        model = SMP_W(decoder=smp.PSPNet, encoder_name='timm-mobilenetv3_large_100', in_channels=in_c, classes=n_classes, encoder_weights=e_ws)
    elif model_name == 'PSPNet_resnet34_W':
        model = SMP_W(decoder=smp.PSPNet, encoder_name='resnet34', in_channels=in_c, classes=n_classes, encoder_weights=e_ws)
    elif model_name == 'PSPNet_resnext50_W':
        model = SMP_W(decoder=smp.PSPNet, encoder_name='resnext50_32x4d', in_channels=in_c, classes=n_classes, encoder_weights=e_ws)
    elif model_name == 'PSPNet_dpn68_W':
        model = SMP_W(decoder=smp.PSPNet, encoder_name='dpn68', in_channels=in_c, classes=n_classes, encoder_weights=e_ws)
    elif model_name == 'PSPNet_dpn92_W':
        model = SMP_W(decoder=smp.PSPNet, encoder_name='dpn92', in_channels=in_c, classes=n_classes, encoder_weights='imagenet+5k')



    elif model_name == 'fpnet_resnext101':
        model = smp.FPN(encoder_name='resnext101_32x4d', in_channels=in_c, classes=n_classes, encoder_weights='ssl')
    elif model_name == 'fpnet_resnext101_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='resnext101_32x4d', in_channels=in_c, classes=n_classes, encoder_weights='ssl')
    elif model_name == 'fpnet_resnext101_8d_W_2':
        model = SMP_W(decoder=smp.FPN, encoder_name='resnext101_32x8d', in_channels=in_c, classes=n_classes, encoder_weights=e_ws)
    elif model_name == 'PSPNet_resnext101_4d_W':
        model = SMP_W(decoder=smp.PSPNet, encoder_name='resnext101_32x4d', in_channels=in_c, classes=n_classes, encoder_weights='ssl')
    elif model_name == 'PSPNet_resnext101_8d_W_2':
        model = SMP_W(decoder=smp.PSPNet, encoder_name='resnext101_32x8d', in_channels=in_c, classes=n_classes, encoder_weights=e_ws)


    elif model_name == 'fpnet_mobilenetv3_large_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='timm-mobilenetv3_large_100', in_channels=in_c, classes=n_classes, encoder_weights=e_ws)

    elif model_name == 'fpnet_se_resnet50_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='se_resnet50', in_channels=in_c, classes=n_classes)
    elif model_name == 'fpnet_skresnet34_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='timm-skresnet34', in_channels=in_c, classes=n_classes)
    elif model_name == 'fpnet_densenet161_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='densenet161', in_channels=in_c, classes=n_classes)


    elif model_name == 'fpnet_resnest26_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='timm-resnest26d', in_channels=in_c, classes=n_classes)
    elif model_name == 'fpnet_resnest50_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='timm-resnest50d', in_channels=in_c, classes=n_classes)
    elif model_name == 'fpnet_res2net50_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='timm-res2net50_26w_4s', in_channels=in_c, classes=n_classes)
    elif model_name == 'fpnet_regnetx40_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='timm-regnetx_040', in_channels=in_c, classes=n_classes)
    elif model_name == 'fpnet_regnety40_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='timm-regnety_040', in_channels=in_c, classes=n_classes)
    elif model_name == 'fpnet_effB4_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='efficientnet-b4', in_channels=in_c, classes=n_classes)

    ########################

    ## DeepLabV3Plus ##
    elif model_name == 'deeplab_resnet18':
        model = smp.DeepLabV3Plus(encoder_name='resnet18', in_channels=in_c, classes=n_classes)
    elif model_name == 'deeplab_resnet18_W':
        model = SMP_W(decoder=smp.DeepLabV3Plus, encoder_name='resnet18', in_channels=in_c, classes=n_classes)

    elif model_name == 'deeplab_mobilenet':
        model = smp.DeepLabV3Plus(encoder_name='mobilenet_v2', in_channels=in_c, classes=n_classes)
    elif model_name == 'deeplab_mobilenet_W':
        model = SMP_W(decoder=smp.DeepLabV3Plus, encoder_name='mobilenet_v2', in_channels=in_c, classes=n_classes)

    elif model_name == 'deeplab_resnet34':
        model = smp.DeepLabV3Plus(encoder_name='resnet34', in_channels=in_c, classes=n_classes)
    elif model_name == 'deeplab_resnet34_W':
        model = SMP_W(decoder=smp.DeepLabV3Plus, encoder_name='resnet34', in_channels=in_c, classes=n_classes)

    elif model_name == 'deeplab_dpn68':
        model = smp.DeepLabV3Plus(encoder_name='dpn68', in_channels=in_c, classes=n_classes)
    elif model_name == 'deeplab_dpn68_W':
        model = SMP_W(decoder=smp.DeepLabV3Plus, encoder_name='dpn68', in_channels=in_c, classes=n_classes)

    elif model_name == 'deeplab_densenet121':
        model = smp.DeepLabV3Plus(encoder_name='densenet121', in_channels=in_c, classes=n_classes)
    elif model_name == 'deeplab_densenet121_W':
        model = SMP_W(decoder=smp.DeepLabV3Plus, encoder_name='densenet121', in_channels=in_c, classes=n_classes)

    else: sys.exit('not a valid model_name, check models.get_model.py')

    setattr(model, 'n_classes', n_classes)

    if 'dpn' in model_name:
        mean, std = [124 / 255, 117 / 255, 104 / 255], [1 / (.0167 * 255)] * 3
    else:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return model, mean, std



