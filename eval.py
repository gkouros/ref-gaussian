import torch
from scene import Scene
import os, time
import numpy as np
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_surfel
import torchvision
from utils.general_utils import safe_state, make_cubemap_faces
from utils.system_utils import searchForMaxIteration
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import get_lpips_model
from torchvision.utils import save_image, make_grid

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

LPIPS = get_lpips_model(net_type='vgg').cuda()

def gamma_tonemap(img, gamma=2.2):
    if isinstance(img, torch.Tensor):
        return torch.clamp(img ** (1.0 / gamma), 0, 1)
    elif isinstance(img, np.ndarray):
        return np.clip(img ** (1.0 / gamma), 0, 1)
    else:
        raise RuntimeWarning(f"gamma_tonemap is not defined for type {type(img)}")

def render_set(model_path, name, views, gaussians, pipeline, background, save_ims, opt, args):
    if save_ims:
        # Create directories to save rendered images
        render_path = os.path.join(model_path, name, "renders")
        gt_path = os.path.join(render_path, 'gt')
        color_path = os.path.join(render_path, 'rgb')
        vis_path = os.path.join(render_path, 'vis')
        makedirs(color_path, exist_ok=True)
        makedirs(gt_path, exist_ok=True)
        makedirs(vis_path, exist_ok=True)

    ssims = []
    psnrs = []
    lpipss = []
    render_times = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        view.refl_mask = None  # When evaluating, reflection mask is disabled
        t1 = time.time()
        
        rendering = render_surfel(view, gaussians, pipeline, background, srgb=opt.srgb, opt=opt)
        render_time = time.time() - t1
        
        render_color = torch.clamp(rendering["render"], 0.0, 1.0)
        render_color = render_color[None]
        gt = torch.clamp(view.original_image, 0.0, 1.0)
        gt = gt[None, 0:3, :, :]
        mask = view.gt_alpha_mask.bool()

        # rescale colors to match with GT
        if args.relight_gt_path and args.relight_envmap_path and args.rescale_relighted:
            if True: # multiscale rescaling
                gt_mean = gt[:, :3, mask.squeeze()].mean(axis=2)
                render_mean = render_color[:, :3, mask.squeeze()].mean(axis=2)
            else:
                gt_mean = gt[:3, mask.squeeze()].mean(None, None, True).squeeze(-1)
                render_mean = render_color[:, :3, mask.squeeze()].mean(None, None, True).squeeze(-1)

            factor = gt_mean / render_mean
            render_color = factor[:, :, None, None] * render_color * mask + (1 - mask.byte())

        ssims.append(ssim(render_color, gt).item())
        psnrs.append(psnr(render_color, gt).item())
        lpipss.append(LPIPS(render_color, gt).item())
        render_times.append(render_time)

        if save_ims:
            torchvision.utils.save_image(gt, os.path.join(gt_path, '{0:05d}.png'.format(idx)))
            torchvision.utils.save_image(render_color, os.path.join(color_path, '{0:05d}.png'.format(idx)))
            torchvision.utils.save_image(rendering['rend_normal'] * 0.5 + 0.5, os.path.join(vis_path, 'normal_{0:05d}.png'.format(idx)))
            torchvision.utils.save_image(rendering["base_color_map"].clamp(0.0, 1.0)[None], os.path.join(vis_path, 'base_color_{0:05d}.png'.format(idx)))
            torchvision.utils.save_image(rendering["roughness_map"].clamp(0.0, 1.0)[None], os.path.join(vis_path, 'roughness_{0:05d}.png'.format(idx)))
            torchvision.utils.save_image(rendering["specular_map"].clamp(0.0, 1.0)[None], os.path.join(vis_path, 'specular_color_{0:05d}.png'.format(idx)))
            torchvision.utils.save_image(rendering["diffuse_map"].clamp(0.0, 1.0)[None], os.path.join(vis_path, 'diffuse_color_{0:05d}.png'.format(idx)))
            torchvision.utils.save_image(rendering["direct_light"].clamp(0.0, 1.0)[None], os.path.join(vis_path, 'direct_light_{0:05d}.png'.format(idx)))
            torchvision.utils.save_image(rendering["indirect_color"].clamp(0.0, 1.0)[None], os.path.join(vis_path, 'indirect_color_{0:05d}.png'.format(idx)))
            torchvision.utils.save_image(rendering["indirect_light"].clamp(0.0, 1.0)[None], os.path.join(vis_path, 'indirect_light_{0:05d}.png'.format(idx)))
            torchvision.utils.save_image(rendering["visibility"].clamp(0.0, 1.0)[None], os.path.join(vis_path, 'visibility_{0:05d}.png'.format(idx)))
            torchvision.utils.save_image(rendering["refl_strength_map"].clamp(0.0, 1.0)[None], os.path.join(vis_path, 'metallic_{0:05d}.png'.format(idx)))
            torchvision.utils.save_image(rendering["rend_alpha"].clamp(0.0, 1.0)[None], os.path.join(vis_path, 'alpha_{0:05d}.png'.format(idx)))
            
    ssim_v = np.array(ssims).mean()
    psnr_v = np.array(psnrs).mean()
    lpip_v = np.array(lpipss).mean()
    fps = 1.0 / np.array(render_times).mean()
    print('psnr:{}, ssim:{}, lpips:{}, fps:{}'.format(psnr_v, ssim_v, lpip_v, fps))
    dump_path = os.path.join(model_path, 'metric.txt')
    with open(dump_path, 'w') as f:
        f.write('psnr:{}, ssim:{}, lpips:{}, fps:{}'.format(psnr_v, ssim_v, lpip_v, fps))

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, save_ims: bool, op, indirect, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        relight = args.relight_gt_path and args.relight_envmap_path
        if relight:
            dataset.source_path = args.relight_gt_path

        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, relight=relight)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        iteration = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
        if indirect:
            op.indirect = 1
            gaussians.load_mesh_from_ply(dataset.model_path, iteration)

        # load relighted envmap
        if relight:
            envmap = cv2.cvtColor(cv2.imread(args.relight_envmap_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
            envmap = cv2.resize(envmap, (1024, 512), interpolation=cv2.INTER_LINEAR)
            envmap = gamma_tonemap(envmap)  # apply tonemap to envmap
            orig_envmap = envmap.copy()
            orig_envmap = np.roll(orig_envmap, shift=envmap.shape[1] // 4, axis=1)
            envmap = envmap * 10 - 5  # expected range of envmap is [-5, 5]
            envmap = np.roll(envmap, shift=envmap.shape[1] // 4, axis=1)
            faces = make_cubemap_faces(envmap, face_size=gaussians.env_map.resolution)
            faces = torch.from_numpy(faces).permute(0,3,1,2).float().cuda()
            fail_value = torch.zeros(gaussians.env_map.output_dim).float().cuda()
            gaussians.env_map.set_faces(faces, fail_value)

        # render_set(dataset.model_path, "train", scene.getTrainCameras(), gaussians, pipeline, background, save_ims, op, args)
        test_dir = "test" if not relight else os.path.join("relight", args.relight_envmap_path.split('/')[-1])
        render_set(dataset.model_path, test_dir, scene.getTestCameras(), gaussians, pipeline, background, save_ims, op, args)
        
        env_dict = gaussians.render_env_map()
        grid = [
            env_dict["env1"].permute(2, 0, 1),
        ]
        grid = make_grid(grid, nrow=1, padding=10)
        save_image(grid, os.path.join(dataset.model_path, "env1.png"))
        grid = [
            env_dict["env2"].permute(2, 0, 1),
        ]
        grid = make_grid(grid, nrow=1, padding=10)
        save_image(grid, os.path.join(dataset.model_path, "env2.png"))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--relight_gt_path", default="")
    parser.add_argument("--relight_envmap_path", default="")
    parser.add_argument("--rescale_relighted", action="store_true")

    # Initialize system state (RNG)
    
    temp_args = parser.parse_args()
    models_path = temp_args.model_path
    exps = os.listdir(models_path)
    for exp in exps:
        args = get_combined_args(parser, exp=exp)
        args.model_path = os.path.join(models_path, exp)
        safe_state(args.quiet)
        print("Rendering " + args.model_path )
        render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.save_images, op, indirect=True, args)
