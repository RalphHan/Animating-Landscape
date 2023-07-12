import gradio as gr
import argparse, os, pickle, sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from predictor import ConditionalMotionNet, ConditionalAppearanceNet
from encoder import define_E
from util import generateLoop, videoWrite, normalize, denormalize
import uuid


class AnimatingLandscape():

    def __init__(self, args):
        self.model_path = args.model_path
        self.model_epoch = args.model_epoch
        self.gpu = int(args.gpu)
        self.outdir_path = args.outdir
        self.t_m = float(args.motion_latent_code)
        self.s_m = float(args.motion_speed)
        self.t_a = float(args.appearance_latent_code)
        self.s_a = float(args.appearance_speed)
        self.t_m = min(1., max(0., self.t_m))
        self.s_m = min(1., max(1e-3, self.s_m))
        self.t_a = min(1., max(0., self.t_a))
        self.s_a = min(1., max(1e-3, self.s_a))
        self.TM = int(args.motion_frame_number)
        self.w, self.h = 256, 256  # Image size for network input
        self.fw, self.fh = None, None  # Output image size
        self.pad = 64  # Reflection padding size for sampling outside of the image

        print('Motion: ')
        self.P_m = ConditionalMotionNet()
        param = torch.load(self.model_path + '/PMNet_weight_' + self.model_epoch + '.pth')
        self.P_m.load_state_dict(param)
        if self.gpu > -1:
            self.P_m.cuda(self.gpu)

        with open(self.model_path + '/codebook_m_' + self.model_epoch + '.pkl', 'rb') as f:
            codebook_m = pickle.load(f) if sys.version_info[0] == 2 else pickle.load(f, encoding='latin1')

        id1 = int(np.floor((len(codebook_m) - 1) * self.t_m))
        id2 = int(np.ceil((len(codebook_m) - 1) * self.t_m))
        z_weight = (len(codebook_m) - 1) * self.t_m - np.floor((len(codebook_m) - 1) * self.t_m)
        self.z_m = (1. - z_weight) * codebook_m[id1:id1 + 1] + z_weight * codebook_m[id2:id2 + 1]
        self.z_m = Variable(torch.from_numpy(self.z_m.astype(np.float32)))
        if self.gpu > -1:
            self.z_m = self.z_m.cuda(self.gpu)
        self.initial_coordinate = np.array([np.meshgrid(np.linspace(-1, 1, self.w + 2 * self.pad),
                                                   np.linspace(-1, 1, self.h + 2 * self.pad), sparse=False)]).astype(
            np.float32)
        self.initial_coordinate = Variable(torch.from_numpy(self.initial_coordinate))
        if self.gpu > -1:
            self.initial_coordinate = self.initial_coordinate.cuda(self.gpu)

    def PredictMotion(self, input_image):
        with torch.no_grad():
            test_img_org = cv2.cvtColor(input_image,cv2.COLOR_RGB2BGR)
            fh, fw = test_img_org.shape[:2]
            test_img = cv2.resize(test_img_org, (self.w, self.h))
            test_input = np.array([normalize(test_img)])
            test_input = Variable(torch.from_numpy(test_input.transpose(0, 3, 1, 2)))
            if self.gpu > -1:
                test_input = test_input.cuda(self.gpu)
            padded_test_input = F.pad(test_input, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
            padded_test_input_large = np.array([normalize(test_img_org)])
            padded_test_input_large = Variable(torch.from_numpy(padded_test_input_large.transpose(0, 3, 1, 2)))
            if self.gpu > -1:
                padded_test_input_large = padded_test_input_large.cuda(self.gpu)
            scaled_pads = (int(self.pad * fh / float(self.h)), int(self.pad * fw / float(self.w)))
            padded_test_input_large = F.pad(padded_test_input_large,
                                            (scaled_pads[1], scaled_pads[1], scaled_pads[0], scaled_pads[0]),
                                            mode='reflect')

            V_m = list()
            V_f = list()
            old_correpondence = None
            for t in range(self.TM):
                sys.stdout.write("\rProcessing frame %d, " % (t + 1))
                sys.stdout.flush()

                flow = self.P_m(test_input, self.z_m)
                flow[:, 0, :, :] = flow[:, 0, :, :] * (self.w / float(self.pad * 2 + self.w))
                flow[:, 1, :, :] = flow[:, 1, :, :] * (self.h / float(self.pad * 2 + self.h))
                flow = F.pad(flow, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
                flow = self.s_m * flow
                correspondence = self.initial_coordinate + flow

                if old_correpondence is not None:
                    correspondence = F.grid_sample(old_correpondence, correspondence.permute(0, 2, 3, 1),
                                                   padding_mode='border', align_corners=True)

                correspondence_large = F.upsample(correspondence,
                                                  size=(fh + scaled_pads[0] * 2, fw + scaled_pads[1] * 2),
                                                  mode='bilinear', align_corners=True)
                y_large = F.grid_sample(padded_test_input_large, correspondence_large.permute(0, 2, 3, 1),
                                        padding_mode='border', align_corners=True)
                outimg = y_large.data.cpu().numpy()[0].transpose(1, 2, 0)
                outimg = denormalize(outimg)
                outimg = outimg[scaled_pads[0]:outimg.shape[0] - scaled_pads[0],
                         scaled_pads[1]:outimg.shape[1] - scaled_pads[1]]
                V_m.append(outimg)

                outflowimg = flow.data.cpu().numpy()[0].transpose(1, 2, 0)
                outflowimg = outflowimg[self.pad:outflowimg.shape[0] - self.pad,
                             self.pad:outflowimg.shape[1] - self.pad]
                mag, ang = cv2.cartToPolar(outflowimg[..., 1], outflowimg[..., 0])
                hsv = np.zeros_like(test_img)
                hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = 255
                outflowimg = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                outflowimg = cv2.resize(outflowimg, (fw, fh))
                V_f.append(outflowimg)

                y = F.grid_sample(padded_test_input, correspondence.permute(0, 2, 3, 1), padding_mode='border',
                                  align_corners=True)
                test_input = y[:, :, self.pad:y.shape[2] - self.pad, self.pad:y.shape[3] - self.pad]
                old_correpondence = correspondence

            V_mloop = generateLoop(V_m)

        return V_mloop, V_f

    def GenerateVideo(self, input_image):
        V_mloop, V_f = self.PredictMotion(input_image)
        the_uuid=str(uuid.uuid4())
        out_path=os.path.join(self.outdir_path,'gradio_videos', '%s.mp4'%the_uuid)
        videoWrite(V_mloop, out_path=out_path)
        return out_path


parser = argparse.ArgumentParser(description='AnimatingLandscape')
parser.add_argument('--model_path', default='./models')
parser.add_argument('--model_epoch', default='5000')
parser.add_argument('--gpu', default=0)
parser.add_argument('--motion_latent_code', '-mz', default=np.random.rand())
parser.add_argument('--motion_speed', '-ms', default=0.2)
parser.add_argument('--appearance_latent_code', '-az', default=np.random.rand())
parser.add_argument('--appearance_speed', '-as', default=0.1)
parser.add_argument('--motion_frame_number', '-mn', default=199)
parser.add_argument('--outdir', '-o', default='./outputs')
args = parser.parse_args()

AS = AnimatingLandscape(args)

demo = gr.Interface(AS.GenerateVideo, [gr.Image(value=cv2.cvtColor(cv2.imread("./inputs/1.png"),cv2.COLOR_BGR2RGB))], [gr.Video(format="mp4",autoplay=True)])
demo.launch(server_name='0.0.0.0',server_port=7863)
