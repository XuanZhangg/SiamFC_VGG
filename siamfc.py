from __future__ import absolute_import, division

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR

from got10k.trackers import Tracker


class SiamFC(nn.Module):

    def __init__(self):
        super(SiamFC, self).__init__()
        self.feature = nn.Sequential(
            # conv1
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv2
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv3
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(384, 256, 3, 1, groups=2))
        self._initialize_weights()

    def forward(self, z, x):
        z = self.feature(z)
        x = self.feature(x)

        # fast cross correlation
        n, c, h, w = x.size()
        x = x.view(1, n * c, h, w)
        out = F.conv2d(x, z, groups=n)
        out = out.view(n, 1, out.size(-2), out.size(-1))

        # adjust the scale of responses
        out = 0.001 * out + 0.0

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, mode='fan_out',
                                     nonlinearity='relu')
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None, **kargs):
        super(TrackerSiamFC, self).__init__(
            name='SiamFC', is_deterministic=True)
        self.cfg = self.parse_args(**kargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        print(self.cuda)
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        print(torch.cuda.get_device_properties(self.device))

        # setup model
        self.net = SiamFC()
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

        # setup optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)

        # setup lr scheduler
        self.lr_scheduler = ExponentialLR(
            self.optimizer, gamma=self.cfg.lr_decay)
        
        # for previous center of mass?
        self.loc = (136, 136)

    def parse_args(self, **kargs):
        # default parameters
        cfg = {
            # inference parameters
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            'adjust_scale': 0.001,
            # train parameters
            'initial_lr': 0.01,
            'lr_decay': 0.8685113737513527,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0}

        for key, val in kargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('GenericDict', cfg.keys())(**cfg)

    def init(self, image, box):
        image = np.asarray(image)

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz

        # exemplar image
        self.avg_color = np.mean(image, axis=(0, 1))
        exemplar_image = self._crop_and_resize(
            image, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            pad_color=self.avg_color)

        # exemplar features
        exemplar_image = torch.from_numpy(exemplar_image).to(
            self.device).permute([2, 0, 1]).unsqueeze(0).float()
        with torch.set_grad_enabled(False):
            self.net.eval()
            self.kernel = self.net.feature(exemplar_image)

    def update(self, image):
        image = np.asarray(image)

        # search images
        instance_images = [self._crop_and_resize(
            image, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            pad_color=self.avg_color) for f in self.scale_factors]
        instance_images = np.stack(instance_images, axis=0)
        instance_images = torch.from_numpy(instance_images).to(
            self.device).permute([0, 3, 1, 2]).float()

        # responses
        with torch.set_grad_enabled(False):
            self.net.eval()
            # instances are CNN features
            instances = self.net.feature(instance_images)
            # responses are the response map; type: torch.Tensor
            responses = F.conv2d(instances, self.kernel) * 0.001
        # squeeze, make it a CPU tensor, then change to a NumPy ndarray
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        # responses.shape: (3, 17, 17), no matter the size of input image
        #print(responses.shape)
        responses = np.stack([cv2.resize(
            t, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC) for t in responses], axis=0)
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty
        # responses.shape: (3, 272, 272), no matter the size of input image
        #print(responses.shape)

        # peak scale
        # choose the best scale/response map among the three
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))
        #print(scale_id)

        # peak location
        # response is a single response map (with the best scale)
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        
        
        # use mean-shift to find loc
        # region of interest: a rectangle, with width 2*half_w and 2*half_h
        # half_w and half_h must be integers
        # The reason why we define half_w and half_h is it's easier to find
        # the other points in region of interest centered around start_point
        # e.g. (start_point[0] + half_w, start_point[1] + half_h
        start_point = self.loc
        
        half_w = 30
        half_h = 30
        # what would be the best choice for these parameters except response?
        loc1 = mean_shift(response, start_point, half_w, half_h)
        '''
        lam = 30
        loc1 = mean_shift_kernel(response, start_point, lam)'''
        
        self.loc = loc1
        
        
        # original loc finding method: find the largest value in response map
        '''response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window'''
        #loc = np.unravel_index(response.argmax(), response.shape)
        # loc is a tuple with 2 int elements
        #print(loc1, loc)
        '''
        # # Non-parametric density estimation, designed by project team
        size = 15
        n2, n3 = response.shape
        X_final, Y_final = 0, 0
        import  math
        X_c,  Y_c = math.floor(n2/4), math.floor(n3/4)
        X_old, Y_old = -1, -1
        counter = 0
        while counter <= 250 and X_c != X_old and Y_c != Y_old:
            X_old, Y_old =  X_c,  Y_c
            counter += 1
            for i in range(max(0, X_c - size), min(n2, X_c + size)):
                for j in range(max(0, Y_c - size), min(n3, Y_c + size)):
                    if response[i][j] > response[X_c][Y_c] :
                        X_c = i
                        Y_c = j

        if response[X_c][Y_c] > response[X_final][Y_final]:
             X_final, Y_final = X_c, Y_c

        X_c, Y_c = math.floor(n2 * 3/ 4), math.floor(n3 * 1 / 4)
        X_old, Y_old = -1, -1
        counter = 0
        while counter <= 250 and X_c != X_old and Y_c != Y_old:
            X_old, Y_old =  X_c,  Y_c
            counter += 1
            for i in range(max(0, X_c - size), min(n2, X_c + size)):
                for j in range(max(0, Y_c - size), min(n3, Y_c + size)):
                    if response[i][j] > response[X_c][Y_c] :
                        X_c = i
                        Y_c = j

        if response[X_c][Y_c] > response[X_final][Y_final]:
            X_final, Y_final = X_c, Y_c

        X_c, Y_c = math.floor(n2 * 1 / 4), math.floor(n3 * 3 / 4)
        X_old, Y_old = -1, -1
        counter = 0
        while counter <= 250 and X_c != X_old and Y_c != Y_old:
            X_old, Y_old =  X_c,  Y_c
            counter += 1
            for i in range(max(0, X_c - size), min(n2, X_c + size)):
                for j in range(max(0, Y_c - size), min(n3, Y_c + size)):
                    if response[i][j] > response[X_c][Y_c] :
                        X_c = i
                        Y_c = j

        if response[X_c][Y_c] > response[X_final][Y_final]:
            X_final, Y_final = X_c, Y_c

        X_c, Y_c = math.floor(n2 * 3 / 4), math.floor(n3 * 3 / 4)
        X_old, Y_old = -1, -1
        counter = 0
        while counter <= 250 and X_c != X_old and Y_c != Y_old:
            X_old, Y_old =  X_c,  Y_c
            counter += 1
            for i in range(max(0, X_c - size), min(n2, X_c + size)):
                for j in range(max(0, Y_c - size), min(n3, Y_c + size)):
                    if response[i][j] > response[X_c][Y_c] :
                        X_c = i
                        Y_c = j

        if response[X_c][Y_c] > response[X_final][Y_final]:
            X_final, Y_final = X_c, Y_c


        loc2 = [X_final, Y_final]
        '''
        
        # locate target center
        disp_in_response = np.array(loc1) - self.upscale_sz // 2
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # update bounding box size
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        #print(scale)
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        # see init() method for what each attri in box represents
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box

    def step(self, batch, backward=True, update_lr=False):
        if backward:
            self.net.train()
            if update_lr:
                self.lr_scheduler.step()
        else:
            self.net.eval()

        z = batch[0].to(self.device)
        x = batch[1].to(self.device)

        with torch.set_grad_enabled(backward):
            responses = self.net(z, x)
            labels, weights = self._create_labels(responses.size())
            loss = F.binary_cross_entropy_with_logits(
                responses, labels, weight=weights, size_average=True)

            if backward:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss.item()

    def _crop_and_resize(self, image, center, size, out_size, pad_color):
        # convert box to corners (0-indexed)
        size = round(size)
        corners = np.concatenate((
            np.round(center - (size - 1) / 2),
            np.round(center - (size - 1) / 2) + size))
        corners = np.round(corners).astype(int)

        # pad image if necessary
        pads = np.concatenate((
            -corners[:2], corners[2:] - image.shape[:2]))
        npad = max(0, int(pads.max()))
        if npad > 0:
            image = cv2.copyMakeBorder(
                image, npad, npad, npad, npad,
                cv2.BORDER_CONSTANT, value=pad_color)

        # crop image patch
        corners = (corners + npad).astype(int)
        patch = image[corners[0]:corners[2], corners[1]:corners[3]]

        # resize to out_size
        patch = cv2.resize(patch, (out_size, out_size))

        return patch

    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels, self.weights

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - w // 2
        y = np.arange(h) - h // 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # pos/neg weights
        pos_num = np.sum(labels == 1)
        neg_num = np.sum(labels == 0)
        weights = np.zeros_like(labels)
        weights[labels == 1] = 0.5 / pos_num
        weights[labels == 0] = 0.5 / neg_num
        weights *= pos_num + neg_num

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        weights = weights.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))
        weights = np.tile(weights, [n, c, 1, 1])

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()
        self.weights = torch.from_numpy(weights).to(self.device).float()

        return self.labels, self.weights

# Description: a mean shift implementation specifically for response maps
# produced from this siamfc implementation
# output: the point in the response map where the region of interest centered
# around it has the largest average value
def mean_shift(response_map, start_point, half_w, half_h):
    # calculate center of mass for the region of interest
    # use the values from response map to weight each point
    # an invalid value, to make sure we successfully get into the loop
    new_point = start_point
    while True:
        center = new_point
        new_point = center_of_mass(response_map, center, half_w, half_h)
        #print('in' + str(start_point) + 'loop')
        #print(center, new_point)
        if center == new_point:
            # converged
            break
    return center
    

def center_of_mass(response_map, center, half_w, half_h):
    x = center[0]
    y = center[1]
    denominator = 0
    x_numer = 0
    y_numer = 0
    # Issue: what if center is at corner s.t. some parts of region of interest
    # are missing
    # Solution (so far): just use the limited amount of points
    # use min(), max() to make sure the range objects have valid values
    for i in range(max(x-half_w, 0), min(x+half_w, response_map.shape[0])):
        for j in range(max(y-half_h, 0), min(y+half_h, response_map.shape[1])):
            weight = response_map[i][j]
            denominator += weight
            x_numer += i * weight
            y_numer += j * weight
    final_x = x_numer / denominator
    final_y = y_numer / denominator
    # Issue: what if center of mass doesn't have integer indices
    # Solution (now): round the coordinates, so that they can be used as the new center point
    #print(round(final_x), round(final_y))
    return (round(final_x), round(final_y))

# Warning: This is very slow!
# applies flat kernel with lambda value lam
# this implies region of interest is a circle as x^2 + y^2 = lam^2
def mean_shift_kernel(response_map, start_point, lam):
    # calculate center of mass for the region of interest
    # use the values from response map to weight each point
    # an invalid value, to make sure we successfully get into the loop
    new_point = start_point
    while True:
        center = new_point
        new_point = kernel_CoM(response_map, center, lam)
        #print('in' + str(start_point) + 'loop')
        #print(center, new_point)
        if center == new_point:
            # converged
            break
    return center

def kernel_CoM(response_map, center, lam):
    denominator = 0
    x_numer = 0
    y_numer = 0
    for i in range(response_map.shape[0]):
        for j in range(response_map.shape[1]):
            if np.linalg.norm((center - np.array((i, j))), 2) <= lam:
                weight = response_map[i][j]
                denominator += weight
                x_numer += i * weight
                y_numer += j * weight
    final_x = x_numer / denominator
    final_y = y_numer / denominator
    return (round(final_x), round(final_y))