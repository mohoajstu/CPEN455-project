import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from layers import *
from utils import check_dir_and_create

class PixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        self.u_stream = nn.ModuleList([
            gated_resnet(nr_filters, down_shifted_conv2d,
                         resnet_nonlinearity, skip_connection=1)
            for _ in range(nr_resnet)
        ])

        self.ul_stream = nn.ModuleList([
            gated_resnet(nr_filters, down_right_shifted_conv2d,
                         resnet_nonlinearity, skip_connection=1)
            for _ in range(nr_resnet)
        ])

    def forward(self, u, ul, label_emb=None):
        """
        If label_emb is not None, we broadcast it to match the shape of (B, nr_filters, H, W).
        Then we inject label_emb into both streams.
        """
        u_list, ul_list = [], []
        for i in range(self.nr_resnet):
            if label_emb is not None:
                bsz, c, h, w = u.size()
                label_broad = label_emb.expand(bsz, c, h, w)

                u = self.u_stream[i](u, a=torch.cat((u, label_broad), dim=1))

                ul = self.ul_stream[i](ul, a=torch.cat((u, label_broad), dim=1))
            else:
                u  = self.u_stream[i](u, a=u)      
                ul = self.ul_stream[i](ul, a=u)

            u_list.append(u)
            ul_list.append(ul)

        return u_list, ul_list


class PixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        self.u_stream  = nn.ModuleList([
            gated_resnet(nr_filters, down_shifted_conv2d,
                         resnet_nonlinearity, skip_connection=1)
            for _ in range(nr_resnet)
        ])

        self.ul_stream = nn.ModuleList([
            gated_resnet(nr_filters, down_right_shifted_conv2d,
                         resnet_nonlinearity, skip_connection=2)
            for _ in range(nr_resnet)
        ])

    def forward(self, u, ul, u_list, ul_list):
        for i in range(self.nr_resnet):
            u_prev  = u_list.pop()
            ul_prev = ul_list.pop()

            u  = self.u_stream[i](u,  a=u_prev)
            ul = self.ul_stream[i](ul, a=torch.cat((u, ul_prev), dim=1))

        return u, ul


class conditional_pixelcnn(nn.Module):
    def __init__(self,
                 nr_resnet=5,
                 nr_filters=80,
                 nr_logistic_mix=10,
                 resnet_nonlinearity='concat_elu',
                 input_channels=3,
                 nr_classes=4,
                 emb_dim=None):
        super(conditional_pixelcnn, self).__init__()

        if resnet_nonlinearity == 'concat_elu':
            self.resnet_nonlinearity = concat_elu
        else:
            raise ValueError('Only concat_elu is currently supported as the nonlinearity.')

        self.nr_filters       = nr_filters
        self.input_channels   = input_channels
        self.nr_logistic_mix  = nr_logistic_mix
        self.nr_classes       = nr_classes
        self.emb_dim          = emb_dim if emb_dim is not None else nr_filters

        if self.nr_classes is not None:
            self.label_emb = nn.Embedding(self.nr_classes, self.emb_dim)

        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad  = nn.ZeroPad2d((0, 0, 1, 0))

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([
            PixelCNNLayer_down(down_nr_resnet[i], nr_filters, self.resnet_nonlinearity)
            for i in range(3)
        ])

        self.up_layers = nn.ModuleList([
            PixelCNNLayer_up(nr_resnet, nr_filters, self.resnet_nonlinearity)
            for _ in range(3)
        ])

        self.downsize_u_stream  = nn.ModuleList([
            down_shifted_conv2d(nr_filters, nr_filters, stride=(2,2))
            for _ in range(2)
        ])
        self.downsize_ul_stream = nn.ModuleList([
            down_right_shifted_conv2d(nr_filters, nr_filters, stride=(2,2))
            for _ in range(2)
        ])
        self.upsize_u_stream    = nn.ModuleList([
            down_shifted_deconv2d(nr_filters, nr_filters, stride=(2,2))
            for _ in range(2)
        ])
        self.upsize_ul_stream   = nn.ModuleList([
            down_right_shifted_deconv2d(nr_filters, nr_filters, stride=(2,2))
            for _ in range(2)
        ])

        self.u_init = down_shifted_conv2d(input_channels + 1,
                                          nr_filters,
                                          filter_size=(2,3),
                                          shift_output_down=True)
        self.ul_init = nn.ModuleList([
            down_shifted_conv2d(input_channels + 1, nr_filters,
                                filter_size=(1,3),
                                shift_output_down=True),
            down_right_shifted_conv2d(input_channels + 1, nr_filters,
                                      filter_size=(2,1),
                                      shift_output_right=True)
        ])

        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)

        self.init_padding = None

    def forward(self, x, label=None, sample=False):
        """ 
        x:      [B, C, H, W]
        label:  [B] (long) if class-conditional
        sample: True if sampling
        """
        if self.init_padding is None or self.init_padding.shape[0] != x.shape[0]:
            xs = x.size()
            padding = x.new_ones(xs[0], 1, xs[2], xs[3])
            self.init_padding = padding

        if sample:
            xs = x.size()
            padding = x.new_ones(xs[0], 1, xs[2], xs[3])
            x = torch.cat([x, padding], dim=1)
        else:
            x = torch.cat([x, self.init_padding], dim=1)

        label_emb = None
        if (self.nr_classes is not None) and (label is not None):
            label_emb = self.label_emb(label)               # [B, emb_dim]
            label_emb = label_emb.unsqueeze(-1).unsqueeze(-1)  # [B, emb_dim, 1, 1]

        u_list  = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]

        for i in range(3):
            if i == 1 and (label_emb is not None):
                u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1],
                                                  label_emb=label_emb)
            else:
                u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1], label_emb=None)

            u_list  += u_out
            ul_list += ul_out

            if i != 2:
                u_list.append(self.downsize_u_stream[i](u_list[-1]))
                ul_list.append(self.downsize_ul_stream[i](ul_list[-1]))

        u  = u_list.pop()
        ul = ul_list.pop()

        for i in range(3):
            u, ul = self.down_layers[i](u, ul, u_list, ul_list)
            if i != 2:
                u  = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

        x_out = self.nin_out(F.elu(ul))
        return x_out


class random_classifier(nn.Module):
    """
    Simple placeholder classifier that ignores input and picks random classes.
    """
    def __init__(self, NUM_CLASSES):
        super(random_classifier, self).__init__()
        self.NUM_CLASSES = NUM_CLASSES
        self.fc = nn.Linear(3, NUM_CLASSES)
        print("Random classifier initialized")

        local_dir = os.path.join(os.path.dirname(__file__), 'models')
        check_dir_and_create(local_dir)
        torch.save(self.state_dict(), os.path.join(local_dir, 'conditional_PixelCNN.pth'))

    def forward(self, x, device):
        return torch.randint(0, self.NUM_CLASSES, (x.shape[0],), device=device)
