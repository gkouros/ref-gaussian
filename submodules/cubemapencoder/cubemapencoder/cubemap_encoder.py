import numpy as np

import torch
import torch.nn as nn
#import better_cubemap
from torch.cuda.amp import custom_bwd, custom_fwd

try:
    import _cubemapencoder as _backend
except ImportError:
    from .backend import _backend

_interp_to_id = {
    'nearest': 0,
    'linear': 1
}

class _cubemap_encode(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    #@custom_fwd
    def forward(ctx, inputs, embeddings, fail_value, interpolation, enable_seamless):

        embeddings = embeddings.contiguous()
        inputs = inputs.contiguous()

        C = embeddings.shape[1]
        L = embeddings.shape[2]
        B = inputs.shape[0]

        outputs = torch.empty([C,B],dtype=embeddings.dtype,device=embeddings.device)

        _backend.cubemap_encode_forward(
            inputs, embeddings, fail_value, outputs, interpolation, enable_seamless,
            B, C, L
        )

        params = torch.tensor([interpolation,enable_seamless], dtype=torch.int64)
        ctx.save_for_backward(inputs, embeddings, params)
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_outputs):
        inputs, embeddings, params = ctx.saved_variables
        grad_outputs = grad_outputs.contiguous()

        C = embeddings.shape[1]
        L = embeddings.shape[2]
        B = inputs.shape[0]

        grad_embeddings = torch.zeros_like(embeddings)
        grad_inputs = torch.empty_like(inputs)
        grad_fail =  torch.zeros([C], dtype=embeddings.dtype, device=embeddings.device)

        _backend.cubemap_encode_backward(
            grad_outputs, inputs, embeddings,
            grad_embeddings, grad_inputs, grad_fail,
            params[0].item(), params[1].item(),
            B, C, L
        )
        return grad_inputs, grad_embeddings, grad_fail, None, None

cubemap_encode = _cubemap_encode.apply

'''
class _mipcubemap_encode(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx,xxx):
        pass # TODO
    @staticmethod
    @custom_bwd
    def backward(ctx,xxx):
        pass

mipcubemap_encode = _mipcubemap_encode.apply
'''

class CubemapEncoder(nn.Module):
    def __init__(self, output_dim = 6, resolution = 256, interpolation='linear'):
        super().__init__()

        self.input_dim = 3
        self.resolution = resolution
        self.output_dim = output_dim

        self.interpolation = interpolation
        self.interp_id = _interp_to_id[interpolation]
        self.seamless = 1

        self.params = nn.ParameterDict({
            'Cubemap_texture': nn.Parameter(torch.rand(6, self.output_dim, resolution, resolution)*10-5), 
            'Cubemap_failv': nn.Parameter(torch.zeros(self.output_dim))
        })
        self.n_elems = 6 * self.output_dim * resolution * resolution + self.output_dim

    def __repr__(self):
        return f"CubemapEncoder: input_dim={self.input_dim} output_dim={self.output_dim} resolution={self.resolution} -> {self.n_elems} interpolation={self.interpolation} seamless={self.seamless}"

    def forward(self, inputs):
        pre_shape = inputs.shape[:-1]
        outputs = cubemap_encode(inputs.reshape(-1,3), self.params['Cubemap_texture'], self.params['Cubemap_failv'], self.interp_id, self.seamless).permute(1,0)
        outputs = torch.sigmoid(outputs).reshape(*pre_shape,3)
        return outputs

    def set_faces(self, faces, fail_value):
        with torch.no_grad():
            self.params = nn.ParameterDict({
                'Cubemap_texture': nn.Parameter(faces),
                'Cubemap_failv': nn.Parameter(fail_value),
            })

class MipCubemapEncoder(nn.Module):
    def __init__(self, num_levels=4, level_dim=6, per_level_scale=4, base_resolution=4,interpolation='linear',concat=True):
        super().__init__()
        # default 4x 16x 64x 256x

        self.input_dim = 3
        self.num_levels = num_levels
        self.level_dim = level_dim # encode channels per level
        self.per_level_scale = per_level_scale
        self.base_resolution = base_resolution
        self.concat = concat
        self.output_dim = num_levels * level_dim if concat else level_dim

        self.interpolation = interpolation
        self.interp_id = _interp_to_id[interpolation]
        self.seamless = 1

        params_list = []
        L = float(base_resolution)
        n_elems = 0
        for ii in range(num_levels):
            iL = int(np.ceil(L))
            param = nn.Parameter(torch.empty(6,self.level_dim,iL,iL))
            n_elems += 6 * self.level_dim * iL * iL
            params_list.append(param)
            L = L * per_level_scale
        #self.params_list = nn.ModuleList(*params_list)
        self.params_list = nn.ParameterList(params_list)

        self.fail_value = nn.Parameter(torch.zeros(self.level_dim))
        n_elems += self.level_dim
        self.n_elems = n_elems # total parameters

        self.reset_parameters()

    def reset_parameters(self):
        std = 1e-4
        for ii in range(self.num_levels):
            self.params_list[ii].data.uniform_(-std,std)

    def __repr__(self):
        return f"MipCubemapEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} base_resolution={self.base_resolution} -> {self.n_elems} per_level_scale={self.per_level_scale:.4f} interpolation={self.interpolation} seamless={self.seamless}"

    def forward(self, inputs):
        outputs = []
        for ii in range(self.num_levels):
            x = cubemap_encode(inputs, self.params_list[ii], self.fail_value, self.interp_id, self.seamless)
            outputs.append(x)
        if self.concat:
            outputs = torch.cat(outputs,dim=0)
        else:
            outputs = sum(outputs)
        return outputs.permute(1,0) # CxN -> NxC

    def set_faces(self, faces, fail_value=None, strict=True):
        """
        Set cubemap faces for all mip levels.

        Args:
            faces:
                - list/tuple of length num_levels with tensors shaped (6, C, H, W), or
                - 5D tensor shaped (L, 6, C, H, W) where L == num_levels.
              C must equal self.level_dim. H and W can differ per level.
            fail_value (optional): tensor shaped (C,) for the per-level encoder.
            strict (bool): if True, validates shapes and counts; raises on mismatch.
        """
        device = self.params_list[0].device
        dtype  = self.params_list[0].dtype

        # Normalize input to a list of level tensors
        if isinstance(faces, torch.Tensor) and faces.dim() == 5:
            if strict and faces.shape[0] != self.num_levels:
                raise ValueError(f"faces has {faces.shape[0]} levels, expected {self.num_levels}")
            faces_list = [faces[i] for i in range(faces.shape[0])]
        elif isinstance(faces, (list, tuple)):
            if strict and len(faces) != self.num_levels:
                raise ValueError(f"faces length {len(faces)} != num_levels {self.num_levels}")
            faces_list = list(faces)
        else:
            raise TypeError("faces must be a list/tuple of tensors or a 5D tensor (L,6,C,H,W)")

        with torch.no_grad():
            for i, f in enumerate(faces_list):
                if not isinstance(f, torch.Tensor):
                    f = torch.as_tensor(f)
                if strict:
                    if f.dim() != 4 or f.shape[0] != 6:
                        raise ValueError(f"Level {i}: expected shape (6,C,H,W), got {tuple(f.shape)}")
                    if f.shape[1] != self.level_dim:
                        raise ValueError(f"Level {i}: C={f.shape[1]} != level_dim={self.level_dim}")
                    if f.shape[2] != f.shape[3]:
                        raise ValueError(f"Level {i}: H and W must be equal, got {f.shape[2]}x{f.shape[3]}")

                # Replace the Parameter at this level
                f = f.to(device=device, dtype=dtype).contiguous()
                self.params_list[i] = nn.Parameter(f)

            if fail_value is not None:
                if not isinstance(fail_value, torch.Tensor):
                    fail_value = torch.as_tensor(fail_value)
                if strict and fail_value.numel() != self.level_dim:
                    raise ValueError(f"fail_value has {fail_value.numel()} elems, expected {self.level_dim}")
                self.fail_value = nn.Parameter(fail_value.to(device=device, dtype=dtype))
