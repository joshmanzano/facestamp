import lpips
import torch
from torch import nn
from torchvision import transforms
import math

import utils

import kornia as K
import random

from DiffJPEG import DiffJPEG

class ChannelDecoder(nn.Module):
    def __init__(self, secret_size):
        super().__init__()
        layers = nn.Sequential(
            nn.Linear(secret_size * 4, secret_size * 3),
            nn.ReLU(),
            nn.Linear(secret_size * 3, secret_size * 2),
            nn.ReLU(),
            nn.Linear(secret_size * 2, secret_size),
            nn.ReLU(),
        )
        self.main = layers

    def forward(self, secret):
        output = self.main(secret)
        return output

class ChannelEncoder(nn.Module):
    def __init__(self, secret_size):
        super().__init__()
        layers = nn.Sequential(
            nn.Linear(secret_size, secret_size * 2),
            nn.ReLU(),
            nn.Linear(secret_size * 2, secret_size * 3),
            nn.ReLU(),
            nn.Linear(secret_size * 3, secret_size * 4),
            nn.ReLU(),
        )
        self.main = layers

    def forward(self, secret):
        output = self.main(secret)
        return output

class ChannelDiscriminator(nn.Module):
    def __init__(self, secret_size):
        super().__init__()
        layers = nn.Sequential(
            nn.Linear(secret_size * 4, secret_size * 3),
            nn.ReLU(),
            nn.Linear(secret_size * 3, secret_size * 2),
            nn.ReLU(),
            nn.Linear(secret_size * 2, secret_size),
            nn.ReLU(),
            nn.Linear(secret_size, 1),
            nn.Sigmoid(),
        )
        self.main = layers

    def forward(self, secret):
        output = self.main(secret)
        return output



class AttackNet(nn.Module):
    def __init__(self, im_height, im_width):
        super().__init__()
        self.im_height = im_height
        self.im_width = im_width
        layers = nn.Sequential(
            nn.ConvTranspose2d(3, 16, kernel_size=3, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, stride=1)
        )

        self.main = layers

    def forward(self, image):
        image = image.reshape(-1, 3, self.im_height, self.im_width)
        return self.main(image)

class EncoderNet(nn.Module):
    def __init__(self, secret_size, im_height, im_width, secret_channels=3, image_channels=3, input_channels=9, mask_residual=True):
        super().__init__()

        self.im_height = im_height
        self.im_width = im_width
        self.secret_size = secret_size
        self.secret_channels = secret_channels 
        self.image_channels = image_channels 
        self.mask_residual = mask_residual

        self.unscaled_s_h = int(im_height / 8) 
        self.unscaled_s_w = int(im_width / 8)

        self.secret_dense = nn.Sequential(
            nn.Linear(secret_size, int(secret_channels * self.unscaled_s_h * self.unscaled_s_w)),
            nn.ReLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU())
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU())
        self.up6= nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.conv6= nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.up7= nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.conv7= nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.up8= nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.conv8= nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.up9= nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.conv9= nn.Sequential(
            nn.Conv2d((32 + 32 + input_channels), 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.residual = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)


    def forward(self, inputs):


        secret, image, mask = inputs
        secret = secret - .5
        image = image - .5

        transform = nn.Sequential(
            transforms.CenterCrop(min([image.shape[0], image.shape[1]])),
            transforms.Resize((self.im_height, self.im_width))
        )

        secret = self.secret_dense(secret)
        secret = secret.reshape(-1, self.secret_channels, self.unscaled_s_h, self.unscaled_s_w)

        # image = transform(image)
        # mask = transform(mask)
        # image = image.reshape(-1, self.image_channels, self.im_height, self.im_width)

        secret_enlarged = nn.Upsample(scale_factor=(8,8))(secret)

        inputs = torch.cat([secret_enlarged, image, mask], dim=-3)

        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        up6 = self.up6(nn.Upsample(scale_factor=(2, 2))(conv5))
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = self.conv6(merge6)
        up7 = self.up7(nn.Upsample(scale_factor=(2, 2))(conv6))
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = self.conv7(merge7)
        up8 = self.up8(nn.Upsample(scale_factor=(2, 2))(conv7))
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = self.conv8(merge8)
        up9 = self.up9(nn.Upsample(scale_factor=(2, 2))(conv8))
        merge9 = torch.cat([conv1, up9, inputs], dim=1)
        conv9 = self.conv9(merge9)
        residual = self.residual(conv9)
        if(self.mask_residual):
            residual = residual * mask

        return residual 

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        # return input.view(input.size(0), -1)
        return input.reshape(input.size(0), -1)

class DecoderNet(nn.Module):
    def __init__(self, secret_size, im_height, im_width):
        super().__init__()
        self.secret_size = secret_size
        self.im_height = im_height 
        self.im_width = im_width 
        flatten_output = int(math.ceil(im_height / 32)) * int(math.ceil(im_width / 32)) * 128
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(flatten_output, int(flatten_output/2)),
            nn.ReLU(),
            nn.Linear(int(flatten_output/2), secret_size),
            nn.Sigmoid()
            )

    def forward(self, image):
        image = image - .5
        image = image.reshape(-1, 3, self.im_height, self.im_width)

        decoded = self.decoder(image)

        return decoded

def distort(args, encoded_image, distortion='none'):
    if(distortion == 'none'):
        return encoded_image
    elif(distortion == 'rw_distortion'):
        augment = torch.nn.Sequential(
            K.augmentation.RandomPerspective(0.5, p=1),
            K.augmentation.RandomMotionBlur(5, 1.0,1.0, p=1, resample = 'bilinear'),
            K.augmentation.RandomGaussianBlur((3, 3), (1, 3.0), p=1.),
            K.augmentation.RandomGaussianNoise(mean=0., std=0.2, p=1.),
            K.augmentation.ColorJitter(0.3, (0.5, 1.5), 0.1, 0.1, p=1.),
        )
        encoded_image = augment(encoded_image).cpu()
        encoded_image = DiffJPEG(args.im_height,args.im_width)(encoded_image)
        encoded_image = encoded_image.cuda()
        return encoded_image
    elif(distortion == 'jpeg_compression'):
        encoded_image = DiffJPEG(args.im_height,args.im_width)(encoded_image)
        encoded_image = encoded_image.cuda()
        return encoded_image
    elif(distortion == 'saturation'):
        return K.augmentation.ColorJitter(saturation=5.0, p=1)(encoded_image)
    elif(distortion == 'hue'):
        return K.augmentation.ColorJitter(hue=0.2, p=1)(encoded_image)
    elif(distortion == 'resize'):
        return K.augmentation.RandomResizedCrop((args.im_height, args.im_width), scale=(0.5, 1.0), p=1)(encoded_image)
    elif(distortion == 'perspective_warp'):
        return K.augmentation.RandomPerspective(0.5, p=1)(encoded_image)
    elif(distortion == 'affine'):
        return K.augmentation.RandomAffine(45, p=1)(encoded_image)
    elif(distortion == 'motion_blur'):
        return K.augmentation.RandomMotionBlur(5, 1.0, 1.0, p=1)(encoded_image)
    elif(distortion == 'color_manipulation'):
        return K.augmentation.ColorJitter( 0.3, ( 0.5, 1.5), 0.1, 0.1, p=1.)(encoded_image)
    elif(distortion == 'gaussian'):
        return K.augmentation.RandomGaussianNoise(mean=0., std=1, p=1.)(encoded_image)
    elif(distortion == 'random_crop'):
        return K.augmentation.RandomCrop((round(args.im_height/2),round(args.im_width/2)), p=1.)(encoded_image)
    elif(distortion == 'grayscale'):
        return K.augmentation.RandomGrayscale(p=1.)(encoded_image)
    elif(distortion == 'crop'):
        return K.augmentation.RandomErasing(p=1.)(encoded_image)
    elif(distortion == 'face_crop'):
        mask = torch.ones(3, args.im_height, args.im_width)
        mask[:,:round(args.im_height/2),:round(args.im_width/2)] = 0
        if(args.cuda):
            mask = mask.cuda()
        return encoded_image * mask
    elif(distortion == 'noise'):
        noise = torch.randn(3, args.im_height, args.im_width)
        if(args.cuda):
            noise = noise.cuda()
        return torch.clip(encoded_image * noise, 0, 1) 
    elif(distortion == 'black'):
        mask = torch.zeros(encoded_image.shape)
        if(args.cuda):
            mask = mask.cuda()
        return torch.clip(encoded_image * mask, 0, 1)
    elif(distortion == 'white'):
        encoded_image = torch.ones(encoded_image.shape)
        if(args.cuda):
            encoded_image = encoded_image.cuda()
        return torch.clip(encoded_image, 0, 1)
    elif(distortion == 'random'):
        encoded_image = torch.randn(encoded_image.shape)
        if(args.cuda):
            encoded_image = encoded_image.cuda()
        return torch.clip(encoded_image, 0, 1)

def eval_model(encoder, decoder, mse, channel_decoder, cos, image_input, mask_input, secret_input, 
                args, region_input):
    
    # this allows the encoder to learn to avoid the facial features
    residual = encoder((secret_input, image_input, mask_input))

    # add mask to residual, to preserve facial features
    # residual = transform_residual(residual, region_input)
    
    encoded_image = image_input + residual 
    encoded_image = torch.clip(encoded_image, min=0, max=1)

    decoded_secret = decoder(encoded_image)
    analyzed_secret = []

    for idx, enc_img in enumerate(encoded_image):
        img = transforms.ToPILImage()(enc_img)
        img_name = utils.save_image(img)
        try:
            analyzed_secret.append(utils.get_secret_string(img_name)[0])
        except Exception as e:
            print(e)
            analyzed_secret.append(utils.random_secret(args.secret_size))

    analyzed_secret = torch.Tensor(analyzed_secret).cuda()

    if(args.channel_coding):
        decoded_secret = channel_decoder(decoded_secret)
        decoded_secret = torch.clip(decoded_secret, 0, 1)
        decoded_secret = torch.round(decoded_secret)
        test_acc = cos(analyzed_secret, decoded_secret).mean()
        secret_loss = mse(decoded_secret, analyzed_secret)
    else:
        decoded_secret = torch.clip(decoded_secret, 0, 1)
        decoded_secret = torch.round(decoded_secret)
        test_acc = cos(analyzed_secret, decoded_secret).mean()
        secret_loss = mse(decoded_secret, analyzed_secret)
    
    return test_acc, secret_loss 

def single_eval(encoder, decoder, channel_decoder, cos, orig_secret_input, secret_input, image_input, mask_input,
                args, region_input):
    
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
    # this allows the encoder to learn to avoid the facial features
        residual = encoder((secret_input, image_input, mask_input))

        # add mask to residual, to preserve facial features
        # residual = transform_residual(residual, region_input)
        
        encoded_image = image_input + residual 
        encoded_image = torch.clip(encoded_image, min=0, max=1)

        decoded_secret = decoder(encoded_image)

        if(args.channel_coding):
            decoded_secret = channel_decoder(decoded_secret)
            decoded_secret = torch.clip(decoded_secret, 0, 1)
            decoded_secret = torch.round(decoded_secret)
            decoded_acc = cos(orig_secret_input, decoded_secret).mean()
        else:
            decoded_secret = torch.clip(decoded_secret, 0, 1)
            decoded_secret = torch.round(decoded_secret)
            decoded_acc = cos(secret_input, decoded_secret).mean()

    encoder.train()
    decoder.train()

    return decoded_acc

def test_model(encoder, decoder, channel_decoder, cos, orig_secret_input, secret_input, image_input, mask_input, 
                args, global_step, writer, region_input, distortions, tag, save_output):
    
    with torch.no_grad():
        residual = encoder((secret_input, image_input, mask_input))

        # add mask to residual, to preserve facial features
        # residual = transform_residual(residual, region_input)
        
        encoded_image = image_input + residual 
        encoded_image = torch.clip(encoded_image, min=0, max=1)

        # distortions = ['saturation','hue','resize','perspective_warp','motion_blur', 'color_manipulation', 'gaussian']

        if save_output:
            final_output = torch.Tensor()
        
        performance_row = []
        for idx, distortion in enumerate(distortions):
            distorted_img = distort(args, encoded_image.clone().detach(), distortion)
            # print(torch.mean(distorted_img), torch.std(distorted_img))
            distorted_decoded = decoder(distorted_img)
            distorted_decoded = torch.clip(distorted_decoded, 0, 1)
            if(args.channel_coding):
                distorted_decoded = channel_decoder(distorted_decoded)
                distorted_decoded = torch.clip(distorted_decoded, 0, 1)
                # distorted_decoded = torch.round(distorted_decoded)
                distorted_acc = cos(orig_secret_input, distorted_decoded).mean()
            else:
                # distorted_decoded = torch.round(distorted_decoded)
                distorted_acc = cos(secret_input, distorted_decoded).mean()
            
            performance_row.append(distorted_acc.item())
            
            if save_output:
                writer.add_scalar(f'{tag}/{distortion}', distorted_acc, global_step)
                distorted_img_single = distorted_img[0]
                d_i_s = utils.draw(distorted_img_single, [('%.2f' % (distorted_acc* 100)) + '%'])
                final_output = torch.cat([final_output, d_i_s], dim=2)

        if save_output:
            writer.add_image(f'evaluation/{tag}', final_output, global_step)

    return performance_row

def build_model(encoder, decoder, channel_decoder, attacker, cos, mse, orig_secret_input, secret_input, image_input, mask_input, 
                args, global_step, writer, region_input, epoch, all_losses):

    # create mask to feed to encoder
    # this allows the encoder to learn to avoid the facial features
    # mask_input = utils.create_mask_input(image_input, region_input)

    residual = encoder((secret_input, image_input, mask_input))

    # add mask to residual, to preserve facial features
    # residual = transform_residual(residual, region_input)
    
    encoded_image = image_input + residual 
    encoded_image = torch.clip(encoded_image, min=0, max=1)

    decoded_secret = decoder(encoded_image)
    decoded_secret = torch.clip(decoded_secret, 0, 1)
    # relu_decoded_secret = torch.nn.ReLU()(decoded_secret.clone().detach())

    # similarity_avg, class_similarity_avg, similarities = utils.get_similarity(orig_secret_input, orig_decoded_secret, sigmoid=False)
    # if(args.channel_coding):
    #     decoded_secret = channel_decoder(decoded_secret)
    #     secret_loss = mse(decoded_secret, orig_secret_input)
    #     similarity = cos(decoded_secret, orig_secret_input).mean()
    # else:
    secret_loss = mse(decoded_secret, secret_input)
    similarity = cos(decoded_secret, secret_input).mean()

    # decoded_stat = utils.get_summary_stat(sig_decoded_secret)
    # secret_stat = utils.get_summary_stat(secret_input)
    # utils.display_summary_stat(decoded_stat)
    # print('---')
    # utils.display_summary_stat(secret_stat)
    # print('###')


    lpips_loss = utils.lpips_loss(image_input, encoded_image)
    residual_loss = (residual ** 2).mean()

    # attack image with distortions to improve robustness (trains encoder and decoder)
    if(not args.distortion_method == 'none'):
        if(args.distortion_method == 'network'):
            adv_image = attacker(encoded_image)
        elif(args.distortion_method == 'rw_distortion'):
            adv_image = distort(args, encoded_image, 'rw_distortion')
        adv_image = torch.clip(adv_image, 0, 1)

        # adv_image_loss = loss_fn.forward(image_input, adv_image).mean()
        if(args.distortion_method == 'network'):
            adv_image_mse_loss = mse(adv_image, image_input)

        adv_decoded_secret = decoder(adv_image)
        adv_decoded_secret = torch.clip(adv_decoded_secret, 0, 1)

        # adv_similarity_avg, adv_class_similarity_avg, adv_similarities = utils.get_similarity(orig_secret_input, orig_adv_decoded_secret, sigmoid=False)
        # if(args.channel_coding):
        #     adv_decoded_secret = channel_decoder(adv_decoded_secret)
        #     adv_similarity = cos(adv_decoded_secret, orig_secret_input).mean()
        #     adv_secret_loss = mse(orig_secret_input, adv_decoded_secret) 
        # else:
        adv_similarity = cos(adv_decoded_secret, secret_input).mean()
        adv_secret_loss = mse(secret_input, adv_decoded_secret) 

        # get adversarial loss
        # adv_loss = adv_image_loss + (args.a1_weight * (1 - (a1_adv_intensity - 1)) * adv_image_mse_loss) - (args.a2_weight * a2_adv_intensity * adv_secret_loss)
        if(args.distortion_method == 'network'):
            adv_loss = (args.a1_weight * adv_image_mse_loss) - (args.a2_weight * adv_secret_loss)
        else:
            adv_loss = (args.a2_weight * adv_secret_loss)
    else:
        adv_loss = None

    lpips_loss = args.lpips_loss_weight * lpips_loss
    secret_loss = args.secret_loss_weight * secret_loss
    residual_loss = args.residual_loss_weight * residual_loss

    if(all_losses):
        if(not args.distortion_method == 'none'):
            loss = lpips_loss + secret_loss + residual_loss + (args.adv_secret_loss_weight * adv_secret_loss)
        else:
            loss = lpips_loss + secret_loss + residual_loss
    else:
        loss = secret_loss
    
    if(adv_loss != None):
        loss += adv_loss

    # if train_acc_avg == None:
    #     train_acc_avg = similarity_avg
    # else:
    #     train_acc_avg += similarity_avg
    #     train_acc_avg /= 2

    # if(not args.distortion_method == 'none'):
    #     if adv_acc_avg == None:
    #         adv_acc_avg = adv_similarity_avg
    #     else:
    #         adv_acc_avg += adv_similarity_avg 
    #         adv_acc_avg /= 2


    # display and saving
    if global_step % args.output_save_interval == 0:
        if(not args.distortion_method == 'none'):
            writer.add_scalar('train_loss/adv_secret_loss', adv_secret_loss, global_step)
            writer.add_scalar('train_metric/adv_acc', adv_similarity, global_step)
        
        # tensorboard logging
        writer.add_scalar('train_loss/lpips_loss', lpips_loss, global_step)
        writer.add_scalar('train_loss/secret_loss', secret_loss, global_step)
        writer.add_scalar('train_loss/residual_loss', residual_loss, global_step)
        writer.add_scalar('train_loss/loss', loss, global_step)
        writer.add_scalar('train_metric/train_acc', similarity, global_step)
        # writer.add_scalar('train_metric/train_acc', similarity_avg, global_step)
        with torch.no_grad():
            if args.verbose:
                print(f'Saved output at {global_step}')
            final_output = torch.Tensor()

            decoded_secret = torch.nn.Sigmoid()(decoded_secret)
            if(args.distortion_method == 'network'):
                adv_decoded_secret = torch.nn.Sigmoid()(adv_decoded_secret)

            for idx, image in enumerate(image_input):
                ss = utils.to_secret_string(secret_input[idx])
                ds = utils.to_secret_string(torch.round(decoded_secret[idx]))
                ss = utils.parse_string(ss)
                ds = utils.parse_string(ds)
                if(args.distortion_method == 'network'):
                    ads = utils.to_secret_string(torch.round(adv_decoded_secret[idx]))
                    ads = utils.parse_string(ads)
                    a_i = utils.draw(adv_image[idx].cpu().clone().detach(), [str(int(adv_similarity* 100)) + '%'] + ads)
                else:
                    a_i = torch.Tensor()
                i = utils.draw(image_input[idx].cpu().clone().detach(), [f'Epoch {str(epoch)}'] + ss)
                r = residual[idx].cpu().clone().detach()
                e_i = utils.draw(encoded_image[idx].cpu().clone().detach(), [str(int(similarity * 100)) + '%'] + ds)
                output = torch.cat([i,r,e_i,a_i], dim=2)
                final_output = torch.cat([final_output, output], dim=1)
            writer.add_image('output/current_output', final_output, global_step)
    
    return loss, similarity


## Test code, ignore

def test():
    secret = torch.rand((4,14))
    image = torch.rand((4,3,512,384))
    mask = torch.rand((4,1,512,384))
    encoder = EncoderNet(secret_size=14, im_height=128, im_width=128)
    encoded = encoder((secret, image, mask))
    print(encoded.shape)
    decoder = DecoderNet(secret_size=14, im_height=128, im_width=128)
    decoded = decoder(encoded)
    print(decoded.shape)

if __name__ == '__main__':
    test()