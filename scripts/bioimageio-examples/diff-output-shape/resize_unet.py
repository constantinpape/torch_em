import torch
import torch.nn as nn


class UNetBase(nn.Module):
    """
    """
    def __init__(
        self,
        encoder,
        base,
        decoder,
        out_conv=None,
        final_activation=None
    ):
        super().__init__()
        if len(encoder) != len(decoder):
            raise ValueError(f"Incompatible depth of encoder (depth={len(encoder)}) and decoder (depth={len(decoder)})")

        self.encoder = encoder
        self.base = base
        self.decoder = decoder

        if out_conv is None:
            self._out_channels = self.decoder.out_channels
        else:
            self._out_channels = out_conv.out_channels
        self.out_conv = out_conv

        self.final_activation = self._get_activation(final_activation)

    @property
    def in_channels(self):
        return self.encoder.in_channels

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def depth(self):
        return len(self.encoder)

    def _get_activation(self, activation):
        return_activation = None
        if activation is None:
            return None
        if isinstance(activation, nn.Module):
            return activation
        if isinstance(activation, str):
            return_activation = getattr(nn, activation, None)
        if return_activation is None:
            raise ValueError(f"Invalid activation: {activation}")
        return return_activation()

    # load encoder / decoder / base states for pretraining
    def load_encoder_state(self, state):
        self.encoder.load_state_dict(state)

    def load_decoder_state(self, state):
        self.decoder.load_state_dict(state)

    def load_base_state(self, state):
        self.base.load_state_dict(state)

    def _apply_default(self, x):
        self.encoder.return_outputs = True
        self.decoder.return_outputs = False

        x, encoder_out = self.encoder(x)
        x = self.base(x)
        x = self.decoder(x, encoder_inputs=encoder_out[::-1])

        if self.out_conv is not None:
            x = self.out_conv(x)

        x = torch.nn.functional.interpolate(x, scale_factor=0.5)
        if self.final_activation is not None:
            x = self.final_activation(x)

        return x

    def forward(self, x):
        out = self._apply_default(x)
        return out


def _update_conv_kwargs(kwargs, scale_factor):
    # if the scale factor is a scalar or all entries are the same we don't need to update the kwargs
    if isinstance(scale_factor, int) or scale_factor.count(scale_factor[0]) == len(scale_factor):
        return kwargs
    else:  # otherwise set anisotropic kernel
        kernel_size = kwargs.get('kernel_size', 3)
        padding = kwargs.get('padding', 1)

        # bail out if kernel size or padding aren't scalars, because it's
        # unclear what to do in this case
        if not (isinstance(kernel_size, int) and isinstance(padding, int)):
            return kwargs

        kernel_size = tuple(1 if factor == 1 else kernel_size for factor in scale_factor)
        padding = tuple(0 if factor == 1 else padding for factor in scale_factor)
        kwargs.update({'kernel_size': kernel_size, 'padding': padding})
        return kwargs


class Encoder(nn.Module):
    def __init__(
        self,
        features,
        scale_factors,
        conv_block_impl,
        pooler_impl,
        anisotropic_kernel=False,
        **conv_block_kwargs
    ):
        super().__init__()
        if len(features) != len(scale_factors) + 1:
            raise ValueError("Incompatible number of features {len(features)} and scale_factors {len(scale_factors)}")

        conv_kwargs = [conv_block_kwargs] * len(scale_factors)
        if anisotropic_kernel:
            conv_kwargs = [_update_conv_kwargs(kwargs, scale_factor)
                           for kwargs, scale_factor in zip(conv_kwargs, scale_factors)]

        self.blocks = nn.ModuleList(
            [conv_block_impl(inc, outc, **kwargs)
             for inc, outc, kwargs in zip(features[:-1], features[1:], conv_kwargs)]
        )
        self.poolers = nn.ModuleList(
            [pooler_impl(factor) for factor in scale_factors]
        )
        self.return_outputs = True

        self.in_channels = features[0]
        self.out_channels = features[-1]

    def __len__(self):
        return len(self.blocks)

    def forward(self, x):
        encoder_out = []
        for block, pooler in zip(self.blocks, self.poolers):
            x = block(x)
            encoder_out.append(x)
            x = pooler(x)

        if self.return_outputs:
            return x, encoder_out
        else:
            return x


class Decoder(nn.Module):
    def __init__(
        self,
        features,
        scale_factors,
        conv_block_impl,
        sampler_impl,
        anisotropic_kernel=False,
        **conv_block_kwargs
    ):
        super().__init__()
        if len(features) != len(scale_factors) + 1:
            raise ValueError("Incompatible number of features {len(features)} and scale_factors {len(scale_factors)}")

        conv_kwargs = [conv_block_kwargs] * len(scale_factors)
        if anisotropic_kernel:
            conv_kwargs = [_update_conv_kwargs(kwargs, scale_factor)
                           for kwargs, scale_factor in zip(conv_kwargs, scale_factors)]

        self.blocks = nn.ModuleList(
            [conv_block_impl(inc, outc, **kwargs)
             for inc, outc, kwargs in zip(features[:-1], features[1:], conv_kwargs)]
        )
        self.samplers = nn.ModuleList(
            [sampler_impl(factor, inc, outc) for factor, inc, outc
             in zip(scale_factors, features[:-1], features[1:])]
        )
        self.return_outputs = False

        self.in_channels = features[0]
        self.out_channels = features[-1]

    def __len__(self):
        return len(self.blocks)

    # FIXME this prevents traces from being valid for other input sizes, need to find
    # a solution to traceable cropping
    def _crop(self, x, shape):
        shape_diff = [(xsh - sh) // 2 for xsh, sh in zip(x.shape, shape)]
        crop = tuple([slice(sd, xsh - sd) for sd, xsh in zip(shape_diff, x.shape)])
        return x[crop]
        # # Implementation with torch.narrow, does not fix the tracing warnings!
        # for dim, (sh, sd) in enumerate(zip(shape, shape_diff)):
        #     x = torch.narrow(x, dim, sd, sh)
        # return x

    def _concat(self, x1, x2):
        return torch.cat([x1, self._crop(x2, x1.shape)], dim=1)

    def forward(self, x, encoder_inputs):
        if len(encoder_inputs) != len(self.blocks):
            raise ValueError(f"Invalid number of encoder_inputs: expect {len(self.blocks)}, got {len(encoder_inputs)}")

        decoder_out = []
        for block, sampler, from_encoder in zip(self.blocks, self.samplers, encoder_inputs):
            x = sampler(x)
            x = block(self._concat(x, from_encoder))
            decoder_out.append(x)

        if self.return_outputs:
            return decoder_out + [x]
        else:
            return x


def get_norm_layer(norm, dim, channels, n_groups=32):
    if norm is None:
        return None
    if norm == 'InstanceNorm':
        return nn.InstanceNorm2d(channels) if dim == 2 else nn.InstanceNorm3d(channels)
    elif norm == 'GroupNorm':
        return nn.GroupNorm(min(n_groups, channels), channels)
    elif norm == 'BatchNorm':
        return nn.BatchNorm2d(channels) if dim == 2 else nn.BatchNorm3d(channels)
    else:
        raise ValueError(f"Invalid norm: expect one of 'InstanceNorm', 'BatchNorm' or 'GroupNorm', got {norm}")


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim,
                 kernel_size=3, padding=1, norm='InstanceNorm'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        conv = nn.Conv2d if dim == 2 else nn.Conv3d

        if norm is None:
            self.block = nn.Sequential(
                conv(in_channels, out_channels,
                     kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                conv(out_channels, out_channels,
                     kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                get_norm_layer(norm, dim, in_channels),
                conv(in_channels, out_channels,
                     kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                get_norm_layer(norm, dim, out_channels),
                conv(out_channels, out_channels,
                     kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.block(x)


class Upsampler(nn.Module):
    def __init__(self, scale_factor,
                 in_channels, out_channels,
                 dim, mode):
        super().__init__()
        self.mode = mode
        self.scale_factor = scale_factor

        conv = nn.Conv2d if dim == 2 else nn.Conv3d
        self.conv = conv(in_channels, out_channels, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor,
                                      mode=self.mode, align_corners=False)
        x = self.conv(x)
        return x


#
# 2d unet implementations
#

class ConvBlock2d(ConvBlock):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, dim=2, **kwargs)


class Upsampler2d(Upsampler):
    def __init__(self, scale_factor,
                 in_channels, out_channels,
                 mode='bilinear'):
        super().__init__(scale_factor, in_channels, out_channels,
                         dim=2, mode=mode)


class ResizeUNet(UNetBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        depth=4,
        initial_features=32,
        gain=2,
        final_activation=None,
        return_side_outputs=False,
        conv_block_impl=ConvBlock2d,
        pooler_impl=nn.MaxPool2d,
        sampler_impl=Upsampler2d,
        **conv_block_kwargs
    ):
        features_encoder = [in_channels] + [initial_features * gain ** i for i in range(depth)]
        features_decoder = [initial_features * gain ** i for i in range(depth + 1)][::-1]
        scale_factors = depth * [2]

        if return_side_outputs:
            if isinstance(out_channels, int) or out_channels is None:
                out_channels = [out_channels] * depth
            if len(out_channels) != depth:
                raise ValueError()
            out_conv = nn.ModuleList(
                [nn.Conv2d(feat, outc, 1) for feat, outc in zip(features_decoder[1:], out_channels)]
            )
        else:
            out_conv = None if out_channels is None else nn.Conv2d(features_decoder[-1], out_channels, 1)

        super().__init__(
            encoder=Encoder(
                features=features_encoder,
                scale_factors=scale_factors,
                conv_block_impl=conv_block_impl,
                pooler_impl=pooler_impl,
                **conv_block_kwargs
            ),
            decoder=Decoder(
                features=features_decoder,
                scale_factors=scale_factors[::-1],
                conv_block_impl=conv_block_impl,
                sampler_impl=sampler_impl,
                **conv_block_kwargs
            ),
            base=conv_block_impl(
                features_encoder[-1], features_encoder[-1] * gain,
                **conv_block_kwargs
            ),
            out_conv=out_conv,
            final_activation=final_activation,
        )
        self.init_kwargs = {'in_channels': in_channels, 'out_channels': out_channels, 'depth': depth,
                            'initial_features': initial_features, 'gain': gain,
                            'final_activation': final_activation, 'return_side_outputs': return_side_outputs,
                            'conv_block_impl': conv_block_impl, 'pooler_impl': pooler_impl,
                            'sampler_impl': sampler_impl, **conv_block_kwargs}
