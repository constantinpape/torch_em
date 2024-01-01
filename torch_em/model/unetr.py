from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import Decoder, ConvBlock2d, Upsampler2d
from .vit import get_vision_transformer, ViT_MAE, ViT_Sam

try:
    from micro_sam.util import get_sam_model
except ImportError:
    get_sam_model = None

try:
    from segment_anything.utils.transforms import ResizeLongestSide
except ImportError:
    ResizeLongestSide = None


#
# UNETR IMPLEMENTATION [Vision Transformer (ViT from MAE / ViT from SAM) + UNet Decoder from `torch_em`]
#


class UNETR(nn.Module):

    def _load_encoder_from_checkpoint(self, backbone, encoder, checkpoint):

        if isinstance(checkpoint, str):
            if backbone == "sam" and isinstance(encoder, str):
                # If we have a SAM encoder, then we first try to load the full SAM Model
                # (using micro_sam) and otherwise fall back on directly loading the encoder state
                # from the checkpoint
                try:
                    _, model = get_sam_model(
                        model_type=encoder,
                        checkpoint_path=checkpoint,
                        return_sam=True
                    )
                    encoder_state = model.image_encoder.state_dict()
                except Exception:
                    # If we have a MAE encoder, then we directly load the encoder state
                    # from the checkpoint.
                    encoder_state = torch.load(checkpoint)

            elif backbone == "mae":
                # vit initialization hints from:
                #     - https://github.com/facebookresearch/mae/blob/main/main_finetune.py#L233-L242
                encoder_state = torch.load(checkpoint)["model"]
                encoder_state = OrderedDict({
                    k: v for k, v in encoder_state.items()
                    if (k != "mask_token" and not k.startswith("decoder"))
                })

                # let's remove the `head` from our current encoder (as the MAE pretrained don't expect it)
                current_encoder_state = self.encoder.state_dict()
                if ("head.weight" in current_encoder_state) and ("head.bias" in current_encoder_state):
                    del self.encoder.head

        else:
            encoder_state = checkpoint

        self.encoder.load_state_dict(encoder_state)

    def __init__(
        self,
        img_size: int = 1024,
        backbone: str = "sam",
        encoder: Optional[Union[nn.Module, str]] = "vit_b",
        decoder: Optional[nn.Module] = None,
        out_channels: int = 1,
        use_sam_stats: bool = False,
        use_mae_stats: bool = False,
        resize_input: bool = False,  # TODO should this be the default???
        encoder_checkpoint: Optional[Union[str, OrderedDict]] = None,
        final_activation: Optional[Union[str, nn.Module]] = None,
        use_skip_connection: bool = True,
        embed_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.use_sam_stats = use_sam_stats
        self.use_mae_stats = use_mae_stats
        self.use_skip_connection = use_skip_connection

        if isinstance(encoder, str):  # "vit_b" / "vit_l" / "vit_h"
            print(f"Using {encoder} from {backbone.upper()}")
            self.encoder = get_vision_transformer(img_size=img_size, backbone=backbone, model=encoder)
            if encoder_checkpoint is not None:
                self._load_encoder_from_checkpoint(backbone, encoder, encoder_checkpoint)

            in_chans = self.encoder.in_chans
            if embed_dim is None:
                embed_dim = self.encoder.embed_dim

        else:  # `nn.Module` ViT backbone
            self.encoder = encoder

            have_neck = False
            for name, _ in self.encoder.named_parameters():
                if name.startswith("neck"):
                    have_neck = True

            if embed_dim is None:
                if have_neck:
                    embed_dim = self.encoder.neck[2].out_channels  # the value is 256
                else:
                    embed_dim = self.encoder.patch_embed.proj.out_channels

            in_chans = self.encoder.patch_embed.proj.in_channels

        if resize_input:
            if ResizeLongestSide is None:
                raise RuntimeError("ResizeLongestSide is not available. Please install segment anything.")
            self.transform = ResizeLongestSide(self.encoder.img_size)
        else:
            self.transform = None

        # parameters for the decoder network
        depth = 3
        initial_features = 64
        gain = 2
        features_decoder = [initial_features * gain ** i for i in range(depth + 1)][::-1]
        scale_factors = depth * [2]
        self.out_channels = out_channels

        if decoder is None:
            self.decoder = Decoder(
                features=features_decoder,
                scale_factors=scale_factors[::-1],
                conv_block_impl=ConvBlock2d,
                sampler_impl=Upsampler2d
            )
        else:
            self.decoder = decoder

        self.z_inputs = ConvBlock2d(in_chans, features_decoder[-1])

        self.base = ConvBlock2d(embed_dim, features_decoder[0])

        self.out_conv = nn.Conv2d(features_decoder[-1], out_channels, 1)

        self.deconv1 = Deconv2DBlock(embed_dim, features_decoder[0])
        self.deconv2 = Deconv2DBlock(features_decoder[0], features_decoder[1])
        self.deconv3 = Deconv2DBlock(features_decoder[1], features_decoder[2])
        self.deconv4 = Deconv2DBlock(features_decoder[2], features_decoder[3])

        self.deconv_out = SingleDeconv2DBlock(features_decoder[-1], features_decoder[-1])

        self.decoder_head = ConvBlock2d(2 * features_decoder[-1], features_decoder[-1])

        self.final_activation = self._get_activation(final_activation)

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

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device

        if self.use_sam_stats:
            pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(1, -1, 1, 1).to(device)
            pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(1, -1, 1, 1).to(device)
        elif self.use_mae_stats:
            # TODO: add mean std from mae experiments (or open up arguments for this)
            raise NotImplementedError
        else:
            pixel_mean = torch.Tensor([0.0, 0.0, 0.0]).view(1, -1, 1, 1).to(device)
            pixel_std = torch.Tensor([1.0, 1.0, 1.0]).view(1, -1, 1, 1).to(device)

        transform = getattr(self, "transform", None)
        if transform is not None:
            x = transform.apply_image_torch(x)
        input_shape = x.shape[-2:]

        x = (x - pixel_mean) / pixel_std
        h, w = x.shape[-2:]
        padh = self.encoder.img_size - h
        padw = self.encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x, input_shape

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.encoder.img_size, self.encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def forward(self, x):
        original_shape = x.shape[-2:]

        # Reshape the inputs to the shape expected by the encoder
        # and normalize the inputs if normalization is part of the model.
        x, input_shape = self.preprocess(x)

        use_skip_connection = getattr(self, "use_skip_connection", True)

        encoder_outputs = self.encoder(x)

        if isinstance(self.encoder, ViT_Sam) or isinstance(self.encoder, ViT_MAE):
            z12, from_encoder = encoder_outputs
        else:
            z12 = encoder_outputs

        if use_skip_connection:
            # TODO: we share the weights in the deconv(s), and should preferably avoid doing that
            from_encoder = from_encoder[::-1]
            z9 = self.deconv1(from_encoder[0])

            z6 = self.deconv1(from_encoder[1])
            z6 = self.deconv2(z6)

            z3 = self.deconv1(from_encoder[2])
            z3 = self.deconv2(z3)
            z3 = self.deconv3(z3)

            z0 = self.z_inputs(x)

        else:
            z9 = self.deconv1(z12)
            z6 = self.deconv2(z9)
            z3 = self.deconv3(z6)
            z0 = self.deconv4(z3)

        updated_from_encoder = [z9, z6, z3]

        x = self.base(z12)
        x = self.decoder(x, encoder_inputs=updated_from_encoder)
        x = self.deconv_out(x)

        x = torch.cat([x, z0], dim=1)
        x = self.decoder_head(x)

        x = self.out_conv(x)
        if self.final_activation is not None:
            x = self.final_activation(x)

        x = self.postprocess_masks(x, input_shape, original_shape)
        return x


#
#  ADDITIONAL FUNCTIONALITIES
#


class SingleDeconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)


class SingleConv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)


class Conv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv2DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Deconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv2DBlock(in_planes, out_planes),
            SingleConv2DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)
