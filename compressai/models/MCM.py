from functools import partial
import warnings

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
from torch.nn import Conv2d, ConvTranspose2d

from compressai.models import CompressionModel
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.layers import conv3x3, subpel_conv3x3
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ops import ste_round

from pytorch_msssim import SSIM


from . import vgg16
import util.util as util


from .base import CompressionModel
from .utils import conv

count = 1

class MCM(CompressionModel):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_pix_loss=False,
        latent_depth=384,
        hyperprior_depth=192,
        num_slices=12,
        visual_tokens = 144
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.latent_depth = latent_depth
        self.vis_tokens = visual_tokens
        print("vis_tokens: %d" % (self.vis_tokens))

        # loss
        self.vgg = vgg16.define_Feature_Net(False, 'vgg16', [2])

        # entropy model
        self.entropy_bottleneck = EntropyBottleneck(hyperprior_depth)
        self.gaussian_conditional = GaussianConditional(None)
        self.num_slices = num_slices
        self.max_support_slices = self.num_slices // 2
        self.frozen_stages = -1

        self.g_a = nn.Sequential(
            Conv2d(768, 704, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            Conv2d(704, 640, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            Conv2d(640, 512, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            Conv2d(512, latent_depth, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
        )

        self.g_s = nn.Sequential(
            ConvTranspose2d(latent_depth, 512, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            ConvTranspose2d(512, 640, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            ConvTranspose2d(640, 704, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            ConvTranspose2d(704, 768, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
        )

        self.h_a = nn.Sequential(
            conv3x3(384, 384),
            nn.GELU(),
            conv3x3(384, 336),
            nn.GELU(),
            conv3x3(336, 288, stride=2),
            nn.GELU(),
            conv3x3(288, 240),
            nn.GELU(),
            conv3x3(240, 192, stride=2),
        )

        self.h_mean_s = nn.Sequential(
            conv3x3(192, 240),
            nn.GELU(),
            subpel_conv3x3(240, 288, 2),
            nn.GELU(),
            conv3x3(288, 336),
            nn.GELU(),
            subpel_conv3x3(336, 384, 2),
            nn.GELU(),
            conv3x3(384, 384),
        )
        self.h_scale_s = nn.Sequential(
            conv3x3(192, 240),
            nn.GELU(),
            subpel_conv3x3(240, 288, 2),
            nn.GELU(),
            conv3x3(288, 336),
            nn.GELU(),
            subpel_conv3x3(336, 384, 2),
            nn.GELU(),
            conv3x3(384, 384),
        )
        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(384 + 32 * min(i, 6), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            ) for i in range(num_slices))
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(384 + 32 * min(i, 6), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            ) for i in range(num_slices))
        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(384 + 32 * min(i + 1, 7), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            ) for i in range(num_slices))

        self._freeze_stages()

        # --------------------------------------------------------------------------
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, vis_num, state_dict):
        net = cls(visual_tokens=vis_num)
        net.load_state_dict(state_dict)
        return net

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, t_scores, s_scores):
        N, L, D = x.shape
        len_keep = self.vis_tokens

        # face:
        # idx = torch.multinomial(torch.Tensor(t_scores / 255 * ((s_scores + 1) / 255)), num_samples=L, replacement=False).cuda()
        # coco:
        idx = torch.multinomial(torch.Tensor(t_scores * (s_scores / 255 + 1)), num_samples=L, replacement=False).cuda()

        ids_keep = idx[:, :len_keep]
        # ids_restore = torch.argsort(idx, dim=1)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        # mask = torch.ones([N, L], device=x.device)
        # mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        # mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, ids_keep

    def forward_encoder(self, x, t_scores, s_scores):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, ids_keep = self.random_masking(x, t_scores, s_scores)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        x = x[:, 1:, :]

        return x, ids_keep

    def forward_decoder(self, x, ids_keep):
        """
        ids_keep: If it is the training stage, you can transfer the ids_restore 
        to improve the training efficiency, and comment the following code.
        In the case of the inference phase, no code changes are required.
        It is also possible to change nothing and train directly.
        """
        x = self.decoder_embed(x)

        '''For training, comment out the code below'''
        noise = torch.rand(ids_keep.shape[0], 256, device=x.device)  # noise in [0, 1]
        ids_all = torch.argsort(noise, dim=1)
        superset = torch.cat([ids_keep, ids_all], dim=1)
        uniset, count = torch.unique(superset[0], sorted=False, return_counts=True)
        mask = (count == 1)
        ids_remove = uniset.masked_select(mask).unsqueeze(0)
        for i in range(1, ids_keep.shape[0]):
            uniset, count = torch.unique(superset[i], sorted=False, return_counts=True)
            mask = (count == 1)
            ids_remove = torch.cat([ids_remove, uniset.masked_select(mask).unsqueeze(0)], dim=0)

        ids_restore = torch.cat([ids_keep, ids_remove], dim=1)
        '''For training, comment out the code above'''

        ids_restore = torch.argsort(ids_restore, dim=1)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)

        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = x_

        # add pos embed
        x = x + self.decoder_pos_embed[:, 1:, :]

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def forward_loss(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        """
        pred_imgs = self.unpatchify(pred)
        l1_loss = nn.L1Loss()(pred_imgs, imgs)
        ssim_loss = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3)

        fake_F2_denorm = util.de_normalise(pred_imgs)
        real_F2_denorm = util.de_normalise(imgs)
        fake_F2_norm = util.normalise_batch(fake_F2_denorm)
        real_F2_norm = util.normalise_batch(real_F2_denorm)
        features_fake_F2 = self.vgg(fake_F2_norm)
        features_real_F2 = self.vgg(real_F2_norm)
        loss_feature = nn.MSELoss()(features_fake_F2.relu2_2, features_real_F2.relu2_2) + nn.MSELoss()(features_fake_F2.relu3_3, features_real_F2.relu3_3)

        return 1 - ssim_loss(pred_imgs, imgs), l1_loss, loss_feature

    def forward(self, imgs, t_scores, s_scores):
        vis_num = self.vis_tokens
        y, ids_restore = self.forward_encoder(imgs, t_scores, s_scores)
        y = y.view(-1, int(vis_num**0.5), int(vis_num**0.5), self.embed_dim).permute(0, 3, 1, 2).contiguous()

        y = self.g_a(y).float()
        y_shape = y.shape[2:]

        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)

            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)

        y_hat = self.g_s(y_hat)
        y_hat = y_hat.permute(0, 2, 3, 1).contiguous().view(-1, vis_num, self.embed_dim)

        pred = self.forward_decoder(y_hat, ids_restore).float()  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred)
        x_hat = self.unpatchify(pred)

        return {
            "loss": loss,
            "likelihoods": {
                "y": y_likelihoods,
                "z": z_likelihoods
            },
            "x_hat": x_hat
        }

    def compress(self, x, t_scores, s_scores):
        vis_num = self.vis_tokens
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn("Inference on GPU is not recommended for the autoregressive "
                          "models (the entropy coder is run sequentially on CPU).")
        y, ids_keep = self.forward_encoder(x, t_scores, s_scores)

        y = y.view(-1, int(vis_num**0.5), int(vis_num**0.5), self.embed_dim).permute(0, 3, 1, 2).contiguous()
        y = self.g_a(y).float()
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)

        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:], "ids_keep": ids_keep}

    def decompress(self, strings, shape, ids_keep):
        vis_num = self.vis_tokens
        assert isinstance(strings, list) and len(strings) == 2

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)

        y_hat = self.g_s(y_hat)

        y_hat = y_hat.permute(0, 2, 3, 1).contiguous().view(-1, vis_num, self.embed_dim)

        x_hat = self.forward_decoder(y_hat, ids_keep).float()
        x_hat = self.unpatchify(x_hat)
        
        global count
        print(count)
        count += 1
        return {"x_hat": x_hat}