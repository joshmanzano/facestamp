""" Images to images face swapping. """

import os
import face_alignment
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image
import torch
import fsgan.data.landmark_transforms as landmark_transforms
import fsgan.utils.utils as utils
from fsgan.utils.bbox_utils import scale_bbox
from fsgan.utils.seg_utils import blend_seg_pred
from fsgan.utils.obj_factory import obj_factory
from fsgan.utils.video_utils import extract_landmarks_bboxes_euler_from_images
from fsgan.models.hopenet import Hopenet


def process_image(fa, img, size=256):
    detected_faces = fa.face_detector.detect_from_image(img.copy())
    if len(detected_faces) != 1:
        return None, None

    preds = fa.get_landmarks(img, detected_faces)
    landmarks = preds[0]
    bbox = detected_faces[0][:4]

    # Convert bounding boxes format from [min, max] to [min, size]
    bbox[2:] = bbox[2:] - bbox[:2] + 1

    return landmarks, bbox


def unnormalize(tensor, mean, std):
    """Normalize a tensor image with mean and standard deviation.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.

    Returns:
        Tensor: Normalized Tensor image.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def tensor2bgr(img_tensor):
    output_img = unnormalize(img_tensor.clone(), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    output_img = output_img.squeeze().permute(1, 2, 0).cpu().numpy()
    output_img = np.round(output_img[:, :, ::-1] * 255).astype('uint8')

    return output_img


def transfer_mask(img1, img2, mask):
    mask = mask.view(mask.shape[0], 1, mask.shape[1], mask.shape[2]).repeat(1, 3, 1, 1).float()
    out = img1 * mask + img2 * (1 - mask)

    return out


def create_pyramid(img, n=1):
    # If input is a list or tuple return it as it is (probably already a pyramid)
    if isinstance(img, (list, tuple)):
        return img

    pyd = [img]
    for i in range(n - 1):
        pyd.append(torch.nn.functional.avg_pool2d(pyd[-1], 3, stride=2, padding=1, count_include_pad=False))

    return pyd


def crop2img(img, crop, bbox):
    scaled_bbox = scale_bbox(bbox)
    scaled_crop = cv2.resize(crop, (scaled_bbox[3], scaled_bbox[2]), interpolation=cv2.INTER_CUBIC)
    left = -scaled_bbox[0] if scaled_bbox[0] < 0 else 0
    top = -scaled_bbox[1] if scaled_bbox[1] < 0 else 0
    right = scaled_bbox[0] + scaled_bbox[2] - img.shape[1] if (scaled_bbox[0] + scaled_bbox[2] - img.shape[1]) > 0 else 0
    bottom = scaled_bbox[1] + scaled_bbox[3] - img.shape[0] if (scaled_bbox[1] + scaled_bbox[3] - img.shape[0]) > 0 else 0
    crop_bbox = np.array([left, top, scaled_bbox[2] - left - right, scaled_bbox[3] - top - bottom])
    scaled_bbox += np.array([left, top, -left - right, -top - bottom])

    out_img = img.copy()
    out_img[scaled_bbox[1]:scaled_bbox[1] + scaled_bbox[3], scaled_bbox[0]:scaled_bbox[0] + scaled_bbox[2]] = \
        scaled_crop[crop_bbox[1]:crop_bbox[1] + crop_bbox[3], crop_bbox[0]:crop_bbox[0] + crop_bbox[2]]

    return out_img


def main(source_path, target_path,
         arch='res_unet_split.MultiScaleResUNet(in_nc=71,out_nc=(3,3),flat_layers=(2,0,2,3),ngf=128)',
         reenactment_model_path='./fsgan/weights/ijbc_msrunet_256_2_0_reenactment_v1.pth',
         seg_model_path='./fsgan/weights/lfw_figaro_unet_256_2_0_segmentation_v1.pth',
         inpainting_model_path='./fsgan/weights/ijbc_msrunet_256_2_0_inpainting_v1.pth',
         blend_model_path='./fsgan/weights/ijbc_msrunet_256_2_0_blending_v1.pth',
         pose_model_path='./fsgan/weights/hopenet_robust_alpha1.pth',
         pil_transforms1=('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)',
                          'landmark_transforms.Pyramids(2)'),
         pil_transforms2=('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)',
                          'landmark_transforms.Pyramids(2)', 'landmark_transforms.LandmarksToHeatmaps'),
         tensor_transforms1=('landmark_transforms.ToTensor()',
                            'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'),
         tensor_transforms2=('landmark_transforms.ToTensor()',
                             'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'),
         output_path=None, min_radius=2.0, crop_size=256, reverse_output=False, verbose=0, output_crop=False,
         display=False):
    torch.set_grad_enabled(False)

    # Initialize models
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    device, gpus = utils.set_device()

    # Load face reenactment model
    Gr = obj_factory(arch).to(device)
    checkpoint = torch.load(reenactment_model_path)
    Gr.load_state_dict(checkpoint['state_dict'])
    Gr.train(False)

    # Load face segmentation model
    if seg_model_path is not None:
        print('Loading face segmentation model: "' + os.path.basename(seg_model_path) + '"...')
        if seg_model_path.endswith('.pth'):
            checkpoint = torch.load(seg_model_path)
            Gs = obj_factory(checkpoint['arch']).to(device)
            Gs.load_state_dict(checkpoint['state_dict'])
        else:
            Gs = torch.jit.load(seg_model_path, map_location=device)
        if Gs is None:
            raise RuntimeError('Failed to load face segmentation model!')
            Gs.eval()
    else:
        Gs = None

    # Load face inpainting model
    if seg_model_path is not None:
        print('Loading face inpainting model: "' + os.path.basename(inpainting_model_path) + '"...')
        if inpainting_model_path.endswith('.pth'):
            checkpoint = torch.load(inpainting_model_path)
            Gi = obj_factory(checkpoint['arch']).to(device)
            Gi.load_state_dict(checkpoint['state_dict'])
        else:
            Gi = torch.jit.load(inpainting_model_path, map_location=device)
        if Gi is None:
            raise RuntimeError('Failed to load face segmentation model!')
        Gi.eval()
    else:
        Gi = None

    # Load face blending model
    checkpoint = torch.load(blend_model_path)
    Gb = obj_factory(checkpoint['arch']).to(device)
    Gb.load_state_dict(checkpoint['state_dict'])
    Gb.train(False)

    # Initialize pose
    Gp = Hopenet().to(device)
    checkpoint = torch.load(pose_model_path)
    Gp.load_state_dict(checkpoint['state_dict'])
    Gp.train(False)

    # Initialize transformations
    pil_transforms1 = obj_factory(pil_transforms1) if pil_transforms1 is not None else []
    pil_transforms2 = obj_factory(pil_transforms2) if pil_transforms2 is not None else []
    tensor_transforms1 = obj_factory(tensor_transforms1) if tensor_transforms1 is not None else []
    tensor_transforms2 = obj_factory(tensor_transforms2) if tensor_transforms2 is not None else []
    img_transforms1 = landmark_transforms.ComposePyramids(pil_transforms1 + tensor_transforms1)
    img_transforms2 = landmark_transforms.ComposePyramids(pil_transforms2 + tensor_transforms2)

    # Extract landmarks, bounding boxes, and euler angles from source video
    source_frame_indices, source_landmarks, source_bboxes, source_eulers = \
        extract_landmarks_bboxes_euler_from_images(source_path, Gp, fa, device=device)
    if source_frame_indices.size == 0:
        raise RuntimeError('No faces were detected in the source image directory: ' + source_path)

    # Extract landmarks, bounding boxes, and euler angles from target video
    target_frame_indices, target_landmarks, target_bboxes, target_eulers = \
        extract_landmarks_bboxes_euler_from_images(target_path, Gp, fa, device=device)
    if target_frame_indices.size == 0:
        raise RuntimeError('No faces were detected in the target image directory: ' + target_path)

    # Parse image paths
    source_img_paths = glob(os.path.join(source_path, '*.jpg'))
    target_img_paths = glob(os.path.join(target_path, '*.jpg'))

    # For each image in the source directory
    source_valid_frame_ind = 0
    for k, source_img_path in tqdm(enumerate(source_img_paths), unit='images', total=len(source_img_paths)):
        if k not in source_frame_indices:
            continue
        source_img_bgr = cv2.imread(source_img_path)
        if source_img_bgr is None:
            continue
        source_img_rgb = source_img_bgr[:, :, ::-1]
        curr_source_tensor, curr_source_landmarks, curr_source_bbox = img_transforms1(
            source_img_rgb, source_landmarks[source_valid_frame_ind], source_bboxes[source_valid_frame_ind])
        source_valid_frame_ind += 1

        for j in range(len(curr_source_tensor)):
            curr_source_tensor[j] = curr_source_tensor[j].to(device)

        # For each image in the target directory
        target_valid_frame_ind = 0
        for i, target_img_path in enumerate(target_img_paths):
            curr_output_name = '_'.join([os.path.splitext(os.path.basename(source_img_path))[0],
                                         os.path.splitext(os.path.basename(target_img_path))[0]]) + '.jpg'
            curr_output_path = os.path.join(output_path, curr_output_name)
            if os.path.isfile(curr_output_path):
                target_valid_frame_ind += 1
                continue
            target_img_bgr = cv2.imread(target_img_path)
            if target_img_bgr is None:
                continue
            if i not in target_frame_indices:
                continue
            target_img_rgb = target_img_bgr[:, :, ::-1]

            curr_target_tensor, curr_target_landmarks, curr_target_bbox = img_transforms2(
                target_img_rgb, target_landmarks[target_valid_frame_ind], target_bboxes[target_valid_frame_ind])
            curr_target_euler = target_eulers[target_valid_frame_ind]
            target_valid_frame_ind += 1

            # Face reenactment
            reenactment_input_tensor = []
            for j in range(len(curr_source_tensor)):
                curr_target_landmarks[j] = curr_target_landmarks[j].to(device)
                reenactment_input_tensor.append(torch.cat((curr_source_tensor[j], curr_target_landmarks[j]), dim=0).unsqueeze(0))
            reenactment_img_tensor, reenactment_seg_tensor = Gr(reenactment_input_tensor)

            # Segment target image
            target_img_tensor = curr_target_tensor[0].unsqueeze(0).to(device)
            target_seg_pred_tensor = Gs(target_img_tensor)
            target_mask_tensor = target_seg_pred_tensor.argmax(1) == 1

            # Remove the background of the aligned face
            aligned_face_mask_tensor = reenactment_seg_tensor.argmax(1) == 1  # face
            aligned_background_mask_tensor = ~aligned_face_mask_tensor
            aligned_img_no_background_tensor = reenactment_img_tensor.clone()
            aligned_img_no_background_tensor.masked_fill_(aligned_background_mask_tensor.unsqueeze(1), -1.0)

            # Complete face
            inpainting_input_tensor = torch.cat((aligned_img_no_background_tensor, target_mask_tensor.unsqueeze(1).float()), dim=1)
            inpainting_input_tensor_pyd = create_pyramid(inpainting_input_tensor, len(curr_target_tensor))
            completion_tensor = Gi(inpainting_input_tensor_pyd)

            # Blend faces
            transfer_tensor = transfer_mask(completion_tensor, target_img_tensor, target_mask_tensor)
            blend_input_tensor = torch.cat(
                (transfer_tensor, target_img_tensor, target_mask_tensor.unsqueeze(1).float()), dim=1)
            blend_input_tensor_pyd = create_pyramid(blend_input_tensor, len(curr_target_tensor))
            blend_tensor = Gb(blend_input_tensor_pyd)

            # Convert back to numpy images
            blend_img = tensor2bgr(blend_tensor)

            # Render
            if verbose == 0:    # Complete pipeline
                render_img = blend_img if output_crop else crop2img(target_img_bgr, blend_img,
                                                                    curr_target_bbox[0].numpy())
            elif verbose == 1:  # Ablation study (crop only)
                # Reenactment only
                reenactment_only_tensor = transfer_mask(reenactment_img_tensor, target_img_tensor,
                                                        aligned_face_mask_tensor & target_mask_tensor)
                reenactment_only_img = tensor2bgr(reenactment_only_tensor)

                # Completion only
                completion_only_img = tensor2bgr(transfer_tensor)

                # Blend only
                transfer_tensor = transfer_mask(aligned_img_no_background_tensor, target_img_tensor, target_mask_tensor)
                blend_input_tensor = torch.cat(
                    (transfer_tensor, target_img_tensor, target_mask_tensor.unsqueeze(1).float()), dim=1)
                blend_input_tensor_pyd = create_pyramid(blend_input_tensor, len(curr_target_tensor))
                blend_tensor = Gb(blend_input_tensor_pyd)
                blend_only_img = tensor2bgr(blend_tensor)

                # Concatenate
                render_img = np.concatenate((reenactment_only_img, completion_only_img, blend_only_img, blend_img), axis=1)
            elif verbose == 2:
                # Convert back to numpy images
                reenactment_img_bgr = tensor2bgr(reenactment_img_tensor)
                reenactment_seg_bgr = tensor2bgr(blend_seg_pred(reenactment_img_tensor, reenactment_seg_tensor))
                target_seg_bgr = tensor2bgr(blend_seg_pred(target_img_tensor, target_seg_pred_tensor))
                aligned_img_no_background_bgr = tensor2bgr(aligned_img_no_background_tensor)
                completion_bgr = tensor2bgr(completion_tensor)
                transfer_bgr = tensor2bgr(transfer_tensor)
                target_cropped_bgr = tensor2bgr(target_img_tensor)

                pose_axis_bgr = draw_axis(np.zeros_like(target_cropped_bgr), curr_target_euler[0], curr_target_euler[1],
                                          curr_target_euler[2])
                render_img1 = np.concatenate((reenactment_img_bgr, reenactment_seg_bgr, target_seg_bgr), axis=1)
                render_img2 = np.concatenate((aligned_img_no_background_bgr, completion_bgr, transfer_bgr), axis=1)
                render_img3 = np.concatenate((pose_axis_bgr, blend_img, target_cropped_bgr), axis=1)
                render_img = np.concatenate((render_img1, render_img2, render_img3), axis=0)
            elif verbose == 3:
                source_cropped_bgr = tensor2bgr(curr_source_tensor[0].unsqueeze(0))
                target_cropped_bgr = tensor2bgr(target_img_tensor)
                render_img = np.concatenate((source_cropped_bgr, target_cropped_bgr, blend_img), axis=1)
            cv2.imwrite(curr_output_path, render_img)
            if display:
                cv2.imshow('render_img', render_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('face_swap_images2images')
    parser.add_argument('source', metavar='DIR',
                        help='path to source images directory')
    parser.add_argument('-t', '--target', metavar='DIR',
                        help='paths to target images directory')
    parser.add_argument('-a', '--arch',
                        default='res_unet_split.MultiScaleResUNet(in_nc=71,out_nc=(3,3),flat_layers=(2,0,2,3),ngf=128)',
                        help='model architecture object')
    parser.add_argument('-rm', '--reenactment_model', default='./fsgan/weights/ijbc_msrunet_256_2_0_reenactment_v1.pth',
                        metavar='PATH', help='path to face reenactment model')
    parser.add_argument('-sm', '--seg_model', default='./fsgan/weights/lfw_figaro_unet_256_2_0_segmentation_v1.pth',
                        metavar='PATH', help='path to face segmentation model')
    parser.add_argument('-im', '--inpainting_model', default='./fsgan/weights/ijbc_msrunet_256_2_0_inpainting_v1.pth',
                        metavar='PATH', help='path to face inpainting model')
    parser.add_argument('-bm', '--blending_model', default='./fsgan/weights/ijbc_msrunet_256_2_0_blending_v1.pth',
                        metavar='PATH', help='path to face blending model')
    parser.add_argument('-pm', '--pose_model', default='./fsgan/weights/hopenet_robust_alpha1.pth', metavar='PATH',
                        help='path to face pose model')
    parser.add_argument('-pt1', '--pil_transforms1', nargs='+', help='first PIL transforms',
                        default=('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)',
                                 'landmark_transforms.Pyramids(2)'))
    parser.add_argument('-pt2', '--pil_transforms2', nargs='+', help='second PIL transforms',
                        default=('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)',
                                 'landmark_transforms.Pyramids(2)', 'landmark_transforms.LandmarksToHeatmaps'))
    parser.add_argument('-tt1', '--tensor_transforms1', nargs='+', help='first tensor transforms',
                        default=('landmark_transforms.ToTensor()',
                                 'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'))
    parser.add_argument('-tt2', '--tensor_transforms2', nargs='+', help='second tensor transforms',
                        default=('landmark_transforms.ToTensor()',
                                 'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'))
    parser.add_argument('-o', '--output', default=None, metavar='DIR',
                        help='output directory')
    parser.add_argument('-mr', '--min_radius', default=2.0, type=float, metavar='F',
                        help='minimum distance between points in the appearance map')
    parser.add_argument('-cs', '--crop_size', default=256, type=int, metavar='N',
                        help='crop size of the images')
    parser.add_argument('-ro', '--reverse_output', action='store_true',
                        help='reverse the output name to be <target>_<source>')
    parser.add_argument('-v', '--verbose', default=0, type=int, metavar='N',
                        help='number of steps between each loss plot')
    parser.add_argument('-oc', '--output_crop', action='store_true',
                        help='output crop around the face')
    parser.add_argument('-d', '--display', action='store_true',
                        help='display the rendering')
    args = parser.parse_args()
    main(args.source, args.target, args.arch, args.reenactment_model, args.seg_model, args.inpainting_model,
         args.blending_model, args.pose_model, args.pil_transforms1, args.pil_transforms2, args.tensor_transforms1,
         args.tensor_transforms2, args.output, args.min_radius, args.crop_size, args.reverse_output, args.verbose,
         args.output_crop, args.display)
