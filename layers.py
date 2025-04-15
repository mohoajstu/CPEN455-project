import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm as wn
from utils import *

def concat_elu(x):
    """ 
    Concatenated ELU doubles the channel dimension 
    from C to 2*C.
    """
    return F.elu(torch.cat([x, -x], dim=1))

class nin(nn.Module):
    """
    1x1 conv implemented with a linear layer. Expects [B, C, H, W].
    """
    def __init__(self, dim_in, dim_out):
        super(nin, self).__init__()
        self.lin_a = wn(nn.Linear(dim_in, dim_out))
        self.dim_out = dim_out

    def forward(self, x):
        # x: [B, C, H, W] -> permute -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.size()

        # flatten
        out = self.lin_a(x.contiguous().view(B*H*W, C))
        # reshape
        out = out.view(B, H, W, self.dim_out)
        # permute back
        return out.permute(0, 3, 1, 2)

class down_shifted_conv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,3), stride=(1,1),
                 shift_output_down=False, norm='weight_norm'):
        super().__init__()
        self.shift_output_down = shift_output_down
        self.norm = norm
        self.pad  = nn.ZeroPad2d((
            (filter_size[1] - 1)//2, 
            (filter_size[1] - 1)//2, 
            filter_size[0] - 1,
            0
        ))
        self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride)
        if norm == 'weight_norm':
            self.conv = wn(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)

        if shift_output_down:
            self.down_shift = lambda x: down_shift(x, pad=nn.ZeroPad2d((0,0,1,0)))

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        if self.norm == 'batch_norm':
            x = self.bn(x)
        if self.shift_output_down:
            x = self.down_shift(x)
        return x

class down_right_shifted_conv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,2), stride=(1,1),
                 shift_output_right=False, norm='weight_norm'):
        super().__init__()
        self.shift_output_right = shift_output_right
        self.norm = norm
        self.pad = nn.ZeroPad2d((
            filter_size[1]-1, 
            0, 
            filter_size[0]-1, 
            0
        ))

        self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride=stride)
        if norm == 'weight_norm':
            self.conv = wn(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)

        if shift_output_right:
            self.right_shift = lambda x: right_shift(x, pad=nn.ZeroPad2d((1,0,0,0)))

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        if self.norm == 'batch_norm':
            x = self.bn(x)
        if self.shift_output_right:
            x = self.right_shift(x)
        return x

class gated_resnet(nn.Module):
    """
    Gated ResNet block with skip_connection=1.
    We'll do exactly one concat_elu per conv, so we only double channels from 40 -> 80 once.
    """
    def __init__(self, num_filters, conv_op, nonlinearity=concat_elu, skip_connection=1):
        super().__init__()
        self.num_filters = num_filters  # <--- Add this line
        self.skip_connection = skip_connection
        self.nonlinearity    = nonlinearity

        # Each conv sees 2*nf in, since concat_elu doubles channels
        self.conv_input = conv_op(2 * num_filters, num_filters)

        if skip_connection > 0:
            # nin_skip expects also 80 in -> 40 out if skip=1
            self.nin_skip = nin(2 * skip_connection * num_filters, num_filters)

        self.dropout  = nn.Dropout2d(0.5)
        self.conv_out = conv_op(2 * num_filters, 2 * num_filters)

    def forward(self, og_x, a=None):
        # 1) Preprocess the main branch.
        x = self.nonlinearity(og_x)   # Doubles channels: from [B, num_filters, ...] to [B, 2*num_filters, ...]
        x = self.conv_input(x)        # Reduces back to [B, num_filters, ...]
        
        # 2) Incorporate the skip connection if provided.
        if self.skip_connection != 0 and a is not None:
            # The nin_skip module was constructed to expect:
            expected_channels = 2 * self.skip_connection * self.num_filters
            # If the input 'a' does not have expected_channels, apply the nonlinearity to double its channels.
            if a.size(1) != expected_channels:
                a_processed = self.nonlinearity(a)
            else:
                a_processed = a
            x += self.nin_skip(a_processed)
        
        # 3) Continue with dropout, a final conv, and gating.
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.conv_out(x)
        a_, b_ = torch.chunk(x, 2, dim=1)
        out = a_ * torch.sigmoid(b_)
        return og_x + out



class down_shifted_deconv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,3), stride=(1,1)):
        super().__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out,
                                            filter_size, stride,
                                            output_padding=1))
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.deconv(x)
        xs = x.size()
        return x[:, :, :(xs[2] - self.filter_size[0] + 1),
                 (self.filter_size[1]-1)//2 : (xs[3] - (self.filter_size[1]-1)//2)]

class down_right_shifted_deconv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,2), stride=(1,1)):
        super().__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out,
                                            filter_size, stride,
                                            output_padding=1))
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.deconv(x)
        xs = x.size()
        x = x[:, :, :(xs[2] - self.filter_size[0] + 1),
              :(xs[3] - self.filter_size[1] + 1)]
        return x