"""STUNet as in the lumiere paper, but on latent space"""

"""
TODO: 
initializing the trainable modules such that they perform nearest-neighbor down- and up- sampling operations 
results with a good starting point (in terms of the loss function)?? (not implemented)
"""

from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from diffusers import UNet2DConditionModel
from einops import rearrange, repeat
from diffusers.models.unet_2d_condition import UNet2DConditionOutput


class STUNet(UNet2DConditionModel):
    def __init__(self, temporal_attn_n_heads=8, temporal_attn_layers=4, freeze_pretrained=True, debug_verbose: bool = False,
                 temporal_sampling: bool = True, *args, **kwargs):
        """
        :param temporal_attn_n_heads: attention heads for the 1d temporal attention
        :param temporal_attn_layers: number of attention layers for the 1d temporal attention
        :param freeze_pretrained: freeze the pretrained 2d UNet layer weights
        :param debug_verbose: print out information like intermediate results for debugging purposes
        :param temporal_sampling: turn off to not perform temporal downsampling, e.g. to test if t2i base model works as expected
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        # Ensure block_out_channels is provided in kwargs
        if 'block_out_channels' not in kwargs:
            raise ValueError("block_out_channels must be provided in the arguments")

        # Extract and potentially use block_out_channels before initializing the superclass
        block_out_channels = kwargs['block_out_channels']

        """inflation blocks"""
        self.inflate_down_blocks = nn.ModuleList([])
        self.temporal_downsample_blocks = nn.ModuleList([])
        self.inflate_mid_blocks = nn.ModuleList([])
        self.inflate_up_blocks = nn.ModuleList([])
        self.temporal_upsample_blocks = nn.ModuleList([])
        """down, conv inflation blocks"""
        for block_out_c in block_out_channels:
            self.inflate_down_blocks.append(ConvInflationBlock(block_out_c, block_out_c, block_out_c))
            self.temporal_downsample_blocks.append(temporal_resizing(block_out_c, sampling='down'))
        """middle, 1d attention layers"""
        for _ in range(temporal_attn_layers):
            self.inflate_mid_blocks.append(TemporalAttention(num_channels=block_out_channels[-1],
                                                             num_heads=temporal_attn_n_heads))
        """up, conv inflation blocks"""
        for block_out_c in block_out_channels[::-1]:
            self.inflate_up_blocks.append(ConvInflationBlock(block_out_c, block_out_c, block_out_c))
            self.temporal_upsample_blocks.append(temporal_resizing(block_out_c, sampling='up'))

        if freeze_pretrained:
            self.freeze_pretrained_layers()

        self.debug_verbose = debug_verbose
        self.temporal_sampling = temporal_sampling

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        """
        :param sample:
        :param timestep:
        :param encoder_hidden_states:
        :param class_labels:
        :param attention_mask:
        :param return_dict:
        :return:
        """
        debug_verbose = self.debug_verbose
        temporal_sampling = self.temporal_sampling

        b, c, d, h, w = sample.shape
        sample = rearrange(sample, 'b c d h w -> (b d) c h w') # combine batch and frame dim for 2d unet processing

        #####################################################
        ## Copy and Paste from UNET2DConditionModel.forward()
        #####################################################
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)


        # 2. pre-process
        sample = self.conv_in(sample)
        #####################################################
        ## Copy and Paste done!
        #####################################################
        """adding conv and 1d attention layers to inflate now!"""
        def prepare_encoder_hidden_states(encoder_hidden_states, num_frames):
            new_hidden_states = repeat(encoder_hidden_states, 'b s e -> b d s e', d=num_frames) # repeat text for each frame
            new_hidden_states = rearrange(new_hidden_states, 'b d s e -> (b d) s e') # combine for 2d unet
            return new_hidden_states

        def prepare_time_embedding(timesteps, batch_size, class_labels=class_labels):
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            new_timesteps = timesteps.expand(batch_size)

            t_emb = self.time_proj(new_timesteps)

            # timesteps does not contain any weights and will always return f32 tensors
            # but time_embedding might actually be running in fp16. so we need to cast here.
            # there might be better ways to encapsulate this.
            t_emb = t_emb.to(dtype=self.dtype)
            emb = self.time_embedding(t_emb)

            if self.class_embedding is not None:
                if class_labels is None:
                    raise ValueError("class_labels should be provided when num_class_embeds > 0")

                if self.config.class_embed_type == "timestep":
                    class_labels = self.time_proj(class_labels)

                class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
                emb = emb + class_emb
            return emb

        # 3. down
        """for each down block, we inflate with a inflate_down_block"""
        down_block_res_samples = (sample,)
        old_w = w
        for downsample_block, inflate_block, sample_block in zip(self.down_blocks, self.inflate_down_blocks, self.temporal_downsample_blocks):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=prepare_time_embedding(timesteps, sample.shape[0]),
                    encoder_hidden_states=prepare_encoder_hidden_states(encoder_hidden_states, num_frames=d),
                    attention_mask=attention_mask,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=prepare_time_embedding(timesteps, sample.shape[0]))

            sample = rearrange(sample, '(b d) c h w -> b c d h w', b=b, d=d)
            inflated_sample = inflate_block(sample)
            sample = sample + inflated_sample
            if debug_verbose:
                print(f"down, old w: {old_w}, new w: {sample.shape[-1]}, current d: {d}")
            if temporal_sampling:
                if sample.shape[-1] != old_w:
                    """downsample at frame level"""
                    sample = sample_block(sample)
                    d = d // 2 # half the number of frames
                    old_w = sample.shape[-1]
            sample = rearrange(sample, 'b c d h w -> (b d) c h w')
            res_samples = res_samples[:-1] + (sample,)

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(
            sample, prepare_time_embedding(timesteps, sample.shape[0]), encoder_hidden_states=prepare_encoder_hidden_states(encoder_hidden_states, num_frames=d), attention_mask=attention_mask
        )
        """inflate with 1d attention layers"""
        sample = rearrange(sample, '(b d) c h w -> b c d h w', b=b, d=d)
        for mid_inflate_block in self.inflate_mid_blocks:
            sample = mid_inflate_block(sample)
        sample = rearrange(sample, 'b c d h w -> (b d) c h w')

        # 5. up
        old_w = sample.shape[-1]
        for i, (inflate_block, upsample_block, sample_block) in enumerate(zip(self.inflate_up_blocks, self.up_blocks, self.temporal_upsample_blocks)):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=prepare_time_embedding(timesteps, sample.shape[0]),
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=prepare_encoder_hidden_states(encoder_hidden_states, num_frames=d),
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=prepare_time_embedding(timesteps, sample.shape[0]), res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

            """inflate up block"""
            sample = rearrange(sample, '(b d) c h w -> b c d h w', b=b, d=d)
            inflated_sample = inflate_block(sample)
            sample = sample + inflated_sample
            if debug_verbose:
                print(f"up, old w: {old_w}, new w: {sample.shape[-1]}, current d: {d}")
            if temporal_sampling:
                if sample.shape[-1] != old_w:
                    """upsample at frame level"""
                    sample = sample_block(sample)
                    d = d * 2 # double the number of frames
                    old_w = sample.shape[-1]

            sample = rearrange(sample, 'b c d h w -> (b d) c h w')

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        sample = rearrange(sample, '(b d) c h w -> b c d h w', b=b, d=d)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)

    def freeze_pretrained_layers(self):
        # Freeze all layers in the model
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze the inflation blocks
        trainable_blocks = [self.inflate_down_blocks, self.temporal_downsample_blocks,
                            self.inflate_up_blocks, self.temporal_upsample_blocks,
                            self.inflate_mid_blocks]
        for blocks in trainable_blocks:
            for block in blocks:
                for param in block.parameters():
                    param.requires_grad = True


def temporal_resizing(channels, sampling):
    if sampling == "down":
        # layer = [nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))]
        layer = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0),
                             bias=False)
    elif sampling == "up":
        layer = nn.ConvTranspose3d(channels, channels, (4, 1, 1), (2, 1, 1), (1, 0, 0), bias=False)
    else:
        raise ValueError(f"Sampling must be either None, 'down' or 'up', but got {sampling}")

    return layer


class ConvInflationBlock(nn.Sequential):
    """Convolution-based Inflation Block in Lumiere, without the pretrained spatial layers"""
    def __init__(self, in_planes: int, out_planes: int, midplanes: int, stride: int = 1, padding: int = 1) -> None:
        layers = [nn.Conv3d(
                in_planes,
                midplanes,
                kernel_size=(1, 3, 3),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=False,
            ),
            nn.BatchNorm3d(midplanes),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                midplanes, out_planes, kernel_size=(3, 1, 1), stride=(stride, 1, 1), padding=(padding, 0, 0), bias=False
            ),
            nn.BatchNorm3d(out_planes),
            nn.SiLU(inplace=True),]

        super().__init__(
            *layers
        )


class TemporalAttention(nn.Module):
    """1D Attention in Attention based inflation block in Lumiere"""
    def __init__(self, num_channels, num_heads=8):
        super(TemporalAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        # Extract original spatial dimensions before rearranging
        batch_size, channels, depth, height, width = x.shape

        # Assuming x is of shape (batch_size, channels, depth, height, width)
        x = rearrange(x, 'b c d h w -> (b h w) d c')
        # Apply attention to the sequence along the temporal dimension
        attn_output, _ = self.attention(x, x, x)  # shape remains (batch_size, depth, channels*height*width)

        # Reshape the output back to the original shape
        attn_output = rearrange(attn_output, '(b h w) d c -> b c d h w', c=channels, h=height, w=width)

        return attn_output


def print_model_info(model):
    total_params = 0
    total_trainable_params = 0

    print("Trainable layers and their parameters:")
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count

        if param.requires_grad:
            print(f"Layer: {name}, Parameters: {param_count}")
            total_trainable_params += param_count

    print(f"\nTotal number of parameters: {total_params/1e6}M")
    print(f"Total number of trainable parameters: {total_trainable_params/1e6}M, "
          f"{round(total_trainable_params/total_params*100, 2)}%")


if __name__ == "__main__":
    """example usage"""
    model_id = "stabilityai/stable-diffusion-2-1"
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    unet_config = dict(unet.config)
    stunet = STUNet(temporal_attn_layers=5, debug_verbose=True, temporal_sampling=False, **unet_config)
    latents = torch.randn(
        (3, unet.in_channels, 16, 16, 16)
    ).to('cuda')
    text_embeddings = torch.randn(3, 77, 1024).to('cuda')  # 1024 is the dim used in sd2.1
    t = torch.tensor(50).long().to('cuda')
    stunet = stunet.to('cuda')
    out = stunet(latents, t, text_embeddings)
    print(out.sample.shape)

    print_model_info(stunet)
