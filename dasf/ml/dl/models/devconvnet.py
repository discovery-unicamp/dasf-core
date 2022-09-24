#!/usr/bin/env python3

import torch
import numpy as np

from torch.nn import MaxUnpool2d
from torch.nn import MaxPool2d, ConvTranspose2d
from torch.nn import Sequential, Conv2d
from torch.nn import BatchNorm2d, ReLU

from torch.nn import functional as F

from torchmetrics import Metric

from pytorch_lightning import LightningModule

from dasf.pipeline import ParameterOperator


class MyAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.idx = 0

    def set_idx(self, idx):
        self.idx = idx

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # update metric states
        pred = preds.detach().max(1)[1].cpu().numpy()
        gt = torch.squeeze(target, 1).cpu().numpy()

        assert pred.shape == gt.shape

        np.save("out/pred_" + str(self.idx) + ".npy", pred)

        self.correct += np.sum(pred == gt)
        self.total += len(gt.flatten())

    def __str__(self):
        ret = self.compute()
        return str(ret)

    def compute(self):
        # compute final result
        return float(self.correct / self.total)


class NNModule(LightningModule):
    def __init__(
        self, n_classes=4, learned_billinear=False, clip=0.1, class_weights=False
    ):
        super().__init__()

        self.learned_billinear = learned_billinear
        self.n_classes = n_classes
        self.clip = clip

        if class_weights:
            self.class_weights = torch.tensor(
                [0.7151, 0.8811, 0.5156, 0.9346, 0.9683, 0.9852], requires_grad=False
            )
        else:
            self.class_weights = None

        self.class_names = [
            "upper_ns",
            "middle_ns",
            "lower_ns",
            "rijnland_chalk",
            "scruff",
            "zechstein",
        ]

    def cross_entropy_loss(self, input, target, weight=None, ignore_index=255):
        """
        Use 255 to fill empty values when padding or doing any augmentation operations
        like rotation.
        """
        target = torch.squeeze(target, dim=1)
        loss = F.cross_entropy(input, target, weight, reduction="sum", ignore_index=255)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), amsgrad=True)

    def training_step(self, batch, batch_idx):
        images, labels = batch

        outputs = self.forward(images)

        loss = self.cross_entropy_loss(
            input=outputs, target=labels, weight=self.class_weights
        )

        # gradient clipping
        if self.clip != 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)

        return loss

    def test_step(self, test_batch, batch_idx):
        images, labels = test_batch

        preds = self(images)

        file_object = open("test.txt", "w")
        file_object.write(str(preds.shape))
        file_object.write(str(labels.shape))
        file_object.close()

        self.accuracy.set_idx(batch_idx)

        self.accuracy(preds, labels)


class TorchPatchDeConvNetModule(NNModule):
    def __init__(
        self, n_classes=4, learned_billinear=False, clip=0.1, class_weights=False
    ):
        super().__init__(n_classes, learned_billinear, clip, class_weights)

        self.unpool = MaxUnpool2d(2, stride=2)
        self.conv_block1 = Sequential(
            # conv1_1
            Conv2d(1, 64, 3, padding=1),
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv1_2
            Conv2d(64, 64, 3, padding=1),
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # pool1
            MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True),
        )
        # it returns outputs and pool_indices_1

        # 48*48

        self.conv_block2 = Sequential(
            # conv2_1
            Conv2d(64, 128, 3, padding=1),
            BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv2_2
            Conv2d(128, 128, 3, padding=1),
            BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # pool2
            MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True),
        )
        # it returns outputs and pool_indices_2

        # 24*24

        self.conv_block3 = Sequential(
            # conv3_1
            Conv2d(128, 256, 3, padding=1),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv3_2
            Conv2d(256, 256, 3, padding=1),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv3_3
            Conv2d(256, 256, 3, padding=1),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # pool3
            MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True),
        )
        # it returns outputs and pool_indices_3

        # 12*12

        self.conv_block4 = Sequential(
            # conv4_1
            Conv2d(256, 512, 3, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv4_2
            Conv2d(512, 512, 3, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv4_3
            Conv2d(512, 512, 3, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # pool4
            MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True),
        )
        # it returns outputs and pool_indices_4

        # 6*6

        self.conv_block5 = Sequential(
            # conv5_1
            Conv2d(512, 512, 3, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv5_2
            Conv2d(512, 512, 3, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv5_3
            Conv2d(512, 512, 3, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # pool5
            MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True),
        )
        # it returns outputs and pool_indices_5

        # 3*3

        self.conv_block6 = Sequential(
            # fc6
            Conv2d(512, 4096, 3),
            # set the filter size and nor padding to make output into 1*1
            BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        # 1*1

        self.conv_block7 = Sequential(
            # fc7
            Conv2d(4096, 4096, 1),
            # set the filter size to make output into 1*1
            BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        self.deconv_block8 = Sequential(
            # fc6-deconv
            ConvTranspose2d(4096, 512, 3, stride=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        # 3*3

        self.unpool_block9 = Sequential(
            # unpool5
            MaxUnpool2d(2, stride=2),
        )
        # usage unpool(output, indices)

        # 6*6

        self.deconv_block10 = Sequential(
            # deconv5_1
            ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv5_2
            ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv5_3
            ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        self.unpool_block11 = Sequential(
            # unpool4
            MaxUnpool2d(2, stride=2),
        )

        # 12*12

        self.deconv_block12 = Sequential(
            # deconv4_1
            ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv4_2
            ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv4_3
            ConvTranspose2d(512, 256, 3, stride=1, padding=1),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        self.unpool_block13 = Sequential(
            # unpool3
            MaxUnpool2d(2, stride=2),
        )

        # 24*24

        self.deconv_block14 = Sequential(
            # deconv3_1
            ConvTranspose2d(256, 256, 3, stride=1, padding=1),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv3_2
            ConvTranspose2d(256, 256, 3, stride=1, padding=1),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv3_3
            ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        self.unpool_block15 = Sequential(
            # unpool2
            MaxUnpool2d(2, stride=2),
        )

        # 48*48

        self.deconv_block16 = Sequential(
            # deconv2_1
            ConvTranspose2d(128, 128, 3, stride=1, padding=1),
            BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv2_2
            ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        self.unpool_block17 = Sequential(
            # unpool1
            MaxUnpool2d(2, stride=2),
        )

        # 96*96

        self.deconv_block18 = Sequential(
            # deconv1_1
            ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv1_2
            ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        self.seg_score19 = Sequential(
            # seg-score
            Conv2d(64, self.n_classes, 1),
        )

        if self.learned_billinear:
            raise NotImplementedError

    def forward(self, x):
        size0 = x.size()
        conv1, indices1 = self.conv_block1(x)
        size1 = conv1.size()
        conv2, indices2 = self.conv_block2(conv1)
        size2 = conv2.size()
        conv3, indices3 = self.conv_block3(conv2)
        size3 = conv3.size()
        conv4, indices4 = self.conv_block4(conv3)
        size4 = conv4.size()
        conv5, indices5 = self.conv_block5(conv4)

        conv6 = self.conv_block6(conv5)
        conv7 = self.conv_block7(conv6)
        conv8 = self.deconv_block8(conv7)
        conv9 = self.unpool(conv8, indices5, output_size=size4)
        conv10 = self.deconv_block10(conv9)
        conv11 = self.unpool(conv10, indices4, output_size=size3)
        conv12 = self.deconv_block12(conv11)
        conv13 = self.unpool(conv12, indices3, output_size=size2)
        conv14 = self.deconv_block14(conv13)
        conv15 = self.unpool(conv14, indices2, output_size=size1)
        conv16 = self.deconv_block16(conv15)
        conv17 = self.unpool(conv16, indices1, output_size=size0)
        conv18 = self.deconv_block18(conv17)
        out = self.seg_score19(conv18)

        return out

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
        ]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())
        i_layer = 0
        # copy convolutional filters from vgg16
        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0] : ranges[idx][1]], conv_block):
                if isinstance(l1, Conv2d) and isinstance(l2, Conv2d):
                    if i_layer == 0:
                        l2.weight.data = (
                            (
                                l1.weight.data[:, 0, :, :]
                                + l1.weight.data[:, 1, :, :]
                                + l1.weight.data[:, 2, :, :]
                            )
                            / 3.0
                        ).view(l2.weight.size())
                        l2.bias.data = l1.bias.data
                        i_layer = i_layer + 1
                    else:
                        assert l1.weight.size() == l2.weight.size()
                        assert l1.bias.size() == l2.bias.size()
                        l2.weight.data = l1.weight.data
                        l2.bias.data = l1.bias.data
                        i_layer = i_layer + 1


class TorchPatchDeConvNetSkipModule(NNModule):
    def __init__(
        self, n_classes=4, learned_billinear=False, clip=0.1, class_weights=False
    ):
        super().__init__(n_classes, learned_billinear, clip, class_weights)

        self.unpool = MaxUnpool2d(2, stride=2)
        self.conv_block1 = Sequential(
            # conv1_1
            Conv2d(1, 64, 3, padding=1),
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv1_2
            Conv2d(64, 64, 3, padding=1),
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # pool1
            MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True),
        )
        # it returns outputs and pool_indices_1

        # 48*48

        self.conv_block2 = Sequential(
            # conv2_1
            Conv2d(64, 128, 3, padding=1),
            BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv2_2
            Conv2d(128, 128, 3, padding=1),
            BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # pool2
            MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True),
        )
        # it returns outputs and pool_indices_2

        # 24*24

        self.conv_block3 = Sequential(
            # conv3_1
            Conv2d(128, 256, 3, padding=1),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv3_2
            Conv2d(256, 256, 3, padding=1),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv3_3
            Conv2d(256, 256, 3, padding=1),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # pool3
            MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True),
        )
        # it returns outputs and pool_indices_3

        # 12*12

        self.conv_block4 = Sequential(
            # conv4_1
            Conv2d(256, 512, 3, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv4_2
            Conv2d(512, 512, 3, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv4_3
            Conv2d(512, 512, 3, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # pool4
            MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True),
        )
        # it returns outputs and pool_indices_4

        # 6*6

        self.conv_block5 = Sequential(
            # conv5_1
            Conv2d(512, 512, 3, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv5_2
            Conv2d(512, 512, 3, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv5_3
            Conv2d(512, 512, 3, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # pool5
            MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True),
        )
        # it returns outputs and pool_indices_5

        # 3*3

        self.conv_block6 = Sequential(
            # fc6
            Conv2d(512, 4096, 3),
            # set the filter size and nor padding to make output into 1*1
            BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        # 1*1

        self.conv_block7 = Sequential(
            # fc7
            Conv2d(4096, 4096, 1),
            # set the filter size to make output into 1*1
            BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        self.deconv_block8 = Sequential(
            # fc6-deconv
            ConvTranspose2d(4096, 512, 3, stride=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        # 3*3

        self.unpool_block9 = Sequential(
            # unpool5
            MaxUnpool2d(2, stride=2),
        )
        # usage unpool(output, indices)

        # 6*6

        self.deconv_block10 = Sequential(
            # deconv5_1
            ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv5_2
            ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv5_3
            ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        self.unpool_block11 = Sequential(
            # unpool4
            MaxUnpool2d(2, stride=2),
        )

        # 12*12

        self.deconv_block12 = Sequential(
            # deconv4_1
            ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv4_2
            ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv4_3
            ConvTranspose2d(512, 256, 3, stride=1, padding=1),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        self.unpool_block13 = Sequential(
            # unpool3
            MaxUnpool2d(2, stride=2),
        )

        # 24*24

        self.deconv_block14 = Sequential(
            # deconv3_1
            ConvTranspose2d(256, 256, 3, stride=1, padding=1),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv3_2
            ConvTranspose2d(256, 256, 3, stride=1, padding=1),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv3_3
            ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        self.unpool_block15 = Sequential(
            # unpool2
            MaxUnpool2d(2, stride=2),
        )

        # 48*48

        self.deconv_block16 = Sequential(
            # deconv2_1
            ConvTranspose2d(128, 128, 3, stride=1, padding=1),
            BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv2_2
            ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        self.unpool_block17 = Sequential(
            # unpool1
            MaxUnpool2d(2, stride=2),
        )

        # 96*96

        self.deconv_block18 = Sequential(
            # deconv1_1
            ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv1_2
            ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        self.seg_score19 = Sequential(
            # seg-score
            Conv2d(64, self.n_classes, 1),
        )

        if self.learned_billinear:
            raise NotImplementedError

    def forward(self, x):
        size0 = x.size()
        conv1, indices1 = self.conv_block1(x)
        size1 = conv1.size()
        conv2, indices2 = self.conv_block2(conv1)
        size2 = conv2.size()
        conv3, indices3 = self.conv_block3(conv2)
        size3 = conv3.size()
        conv4, indices4 = self.conv_block4(conv3)
        size4 = conv4.size()
        conv5, indices5 = self.conv_block5(conv4)

        conv6 = self.conv_block6(conv5)
        conv7 = self.conv_block7(conv6)
        conv8 = self.deconv_block8(conv7) + conv5
        conv9 = self.unpool(conv8, indices5, output_size=size4)
        conv10 = self.deconv_block10(conv9) + conv4
        conv11 = self.unpool(conv10, indices4, output_size=size3)
        conv12 = self.deconv_block12(conv11) + conv3
        conv13 = self.unpool(conv12, indices3, output_size=size2)
        conv14 = self.deconv_block14(conv13) + conv2
        conv15 = self.unpool(conv14, indices2, output_size=size1)
        conv16 = self.deconv_block16(conv15) + conv1
        conv17 = self.unpool(conv16, indices1, output_size=size0)
        conv18 = self.deconv_block18(conv17)
        out = self.seg_score19(conv18)

        return out

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
        ]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())
        i_layer = 0
        # copy convolutional filters from vgg16
        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0] : ranges[idx][1]], conv_block):
                if isinstance(l1, Conv2d) and isinstance(l2, Conv2d):
                    if i_layer == 0:
                        l2.weight.data = (
                            (
                                l1.weight.data[:, 0, :, :]
                                + l1.weight.data[:, 1, :, :]
                                + l1.weight.data[:, 2, :, :]
                            )
                            / 3.0
                        ).view(l2.weight.size())
                        l2.bias.data = l1.bias.data
                        i_layer = i_layer + 1
                    else:
                        assert l1.weight.size() == l2.weight.size()
                        assert l1.bias.size() == l2.bias.size()
                        l2.weight.data = l1.weight.data
                        l2.bias.data = l1.bias.data
                        i_layer = i_layer + 1


class TorchSectionDeConvNetModule(NNModule):
    def __init__(
        self, n_classes=4, learned_billinear=False, clip=0.1, class_weights=False
    ):
        super().__init__(n_classes, learned_billinear, clip, class_weights)

        self.unpool = MaxUnpool2d(2, stride=2)
        self.conv_block1 = Sequential(
            # conv1_1
            Conv2d(1, 64, 3, padding=1),
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv1_2
            Conv2d(64, 64, 3, padding=1),
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # pool1
            MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True),
        )
        # it returns outputs and pool_indices_1

        # 48*48

        self.conv_block2 = Sequential(
            # conv2_1
            Conv2d(64, 128, 3, padding=1),
            BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv2_2
            Conv2d(128, 128, 3, padding=1),
            BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # pool2
            MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True),
        )
        # it returns outputs and pool_indices_2

        # 24*24

        self.conv_block3 = Sequential(
            # conv3_1
            Conv2d(128, 256, 3, padding=1),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv3_2
            Conv2d(256, 256, 3, padding=1),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv3_3
            Conv2d(256, 256, 3, padding=1),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # pool3
            MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True),
        )
        # it returns outputs and pool_indices_3

        # 12*12

        self.conv_block4 = Sequential(
            # conv4_1
            Conv2d(256, 512, 3, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv4_2
            Conv2d(512, 512, 3, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv4_3
            Conv2d(512, 512, 3, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # pool4
            MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True),
        )
        # it returns outputs and pool_indices_4

        # 6*6

        self.conv_block5 = Sequential(
            # conv5_1
            Conv2d(512, 512, 3, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv5_2
            Conv2d(512, 512, 3, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv5_3
            Conv2d(512, 512, 3, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # pool5
            MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True),
        )
        # it returns outputs and pool_indices_5

        # 3*3

        self.conv_block6 = Sequential(
            # fc6
            Conv2d(512, 4096, 3),
            # set the filter size and nor padding to make output into 1*1
            BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        # 1*1

        self.conv_block7 = Sequential(
            # fc7
            Conv2d(4096, 4096, 1),
            # set the filter size to make output into 1*1
            BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        self.deconv_block8 = Sequential(
            # fc6-deconv
            ConvTranspose2d(4096, 512, 3, stride=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        # 3*3

        self.unpool_block9 = Sequential(
            # unpool5
            MaxUnpool2d(2, stride=2),
        )
        # usage unpool(output, indices)

        # 6*6

        self.deconv_block10 = Sequential(
            # deconv5_1
            ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv5_2
            ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv5_3
            ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        self.unpool_block11 = Sequential(
            # unpool4
            MaxUnpool2d(2, stride=2),
        )

        # 12*12

        self.deconv_block12 = Sequential(
            # deconv4_1
            ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv4_2
            ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv4_3
            ConvTranspose2d(512, 256, 3, stride=1, padding=1),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        self.unpool_block13 = Sequential(
            # unpool3
            MaxUnpool2d(2, stride=2),
        )

        # 24*24

        self.deconv_block14 = Sequential(
            # deconv3_1
            ConvTranspose2d(256, 256, 3, stride=1, padding=1),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv3_2
            ConvTranspose2d(256, 256, 3, stride=1, padding=1),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv3_3
            ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        self.unpool_block15 = Sequential(
            # unpool2
            MaxUnpool2d(2, stride=2),
        )

        # 48*48

        self.deconv_block16 = Sequential(
            # deconv2_1
            ConvTranspose2d(128, 128, 3, stride=1, padding=1),
            BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv2_2
            ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        self.unpool_block17 = Sequential(
            # unpool1
            MaxUnpool2d(2, stride=2),
        )

        # 96*96

        self.deconv_block18 = Sequential(
            # deconv1_1
            ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv1_2
            ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        self.seg_score19 = Sequential(
            # seg-score
            Conv2d(64, self.n_classes, 1),
        )

        if self.learned_billinear:
            raise NotImplementedError

    def forward(self, x):
        size0 = x.size()
        conv1, indices1 = self.conv_block1(x)
        size1 = conv1.size()
        conv2, indices2 = self.conv_block2(conv1)
        size2 = conv2.size()
        conv3, indices3 = self.conv_block3(conv2)
        size3 = conv3.size()
        conv4, indices4 = self.conv_block4(conv3)
        size4 = conv4.size()
        conv5, indices5 = self.conv_block5(conv4)

        conv6 = self.conv_block6(conv5)
        conv7 = self.conv_block7(conv6)
        conv8 = self.deconv_block8(conv7)
        conv9 = self.unpool(conv8, indices5, output_size=size4)
        conv10 = self.deconv_block10(conv9)
        conv11 = self.unpool(conv10, indices4, output_size=size3)
        conv12 = self.deconv_block12(conv11)
        conv13 = self.unpool(conv12, indices3, output_size=size2)
        conv14 = self.deconv_block14(conv13)
        conv15 = self.unpool(conv14, indices2, output_size=size1)
        conv16 = self.deconv_block16(conv15)
        conv17 = self.unpool(conv16, indices1, output_size=size0)
        conv18 = self.deconv_block18(conv17)
        out = self.seg_score19(conv18)

        return out

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
        ]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())
        i_layer = 0
        # copy convolutional filters from vgg16
        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0] : ranges[idx][1]], conv_block):
                if isinstance(l1, Conv2d) and isinstance(l2, Conv2d):
                    if i_layer == 0:
                        l2.weight.data = (
                            (
                                l1.weight.data[:, 0, :, :]
                                + l1.weight.data[:, 1, :, :]
                                + l1.weight.data[:, 2, :, :]
                            )
                            / 3.0
                        ).view(l2.weight.size())
                        l2.bias.data = l1.bias.data
                        i_layer = i_layer + 1
                    else:
                        assert l1.weight.size() == l2.weight.size()
                        assert l1.bias.size() == l2.bias.size()
                        l2.weight.data = l1.weight.data
                        l2.bias.data = l1.bias.data
                        i_layer = i_layer + 1


class TorchSectionDeConvNetSkipModule(NNModule):
    def __init__(
        self, n_classes=4, learned_billinear=False, clip=0.1, class_weights=False
    ):
        super().__init__(n_classes, learned_billinear, clip, class_weights)

        self.unpool = MaxUnpool2d(2, stride=2)
        self.conv_block1 = Sequential(
            # conv1_1
            Conv2d(1, 64, 3, padding=1),
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv1_2
            Conv2d(64, 64, 3, padding=1),
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # pool1
            MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True),
        )
        # it returns outputs and pool_indices_1

        # 48*48

        self.conv_block2 = Sequential(
            # conv2_1
            Conv2d(64, 128, 3, padding=1),
            BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv2_2
            Conv2d(128, 128, 3, padding=1),
            BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # pool2
            MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True),
        )
        # it returns outputs and pool_indices_2

        # 24*24

        self.conv_block3 = Sequential(
            # conv3_1
            Conv2d(128, 256, 3, padding=1),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv3_2
            Conv2d(256, 256, 3, padding=1),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv3_3
            Conv2d(256, 256, 3, padding=1),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # pool3
            MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True),
        )
        # it returns outputs and pool_indices_3

        # 12*12

        self.conv_block4 = Sequential(
            # conv4_1
            Conv2d(256, 512, 3, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv4_2
            Conv2d(512, 512, 3, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv4_3
            Conv2d(512, 512, 3, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # pool4
            MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True),
        )
        # it returns outputs and pool_indices_4

        # 6*6

        self.conv_block5 = Sequential(
            # conv5_1
            Conv2d(512, 512, 3, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv5_2
            Conv2d(512, 512, 3, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # conv5_3
            Conv2d(512, 512, 3, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # pool5
            MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True),
        )
        # it returns outputs and pool_indices_5

        # 3*3

        self.conv_block6 = Sequential(
            # fc6
            Conv2d(512, 4096, 3),
            # set the filter size and nor padding to make output into 1*1
            BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        # 1*1

        self.conv_block7 = Sequential(
            # fc7
            Conv2d(4096, 4096, 1),
            # set the filter size to make output into 1*1
            BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        self.deconv_block8 = Sequential(
            # fc6-deconv
            ConvTranspose2d(4096, 512, 3, stride=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        # 3*3

        self.unpool_block9 = Sequential(
            # unpool5
            MaxUnpool2d(2, stride=2),
        )
        # usage unpool(output, indices)

        # 6*6

        self.deconv_block10 = Sequential(
            # deconv5_1
            ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv5_2
            ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv5_3
            ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        self.unpool_block11 = Sequential(
            # unpool4
            MaxUnpool2d(2, stride=2),
        )

        # 12*12

        self.deconv_block12 = Sequential(
            # deconv4_1
            ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv4_2
            ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv4_3
            ConvTranspose2d(512, 256, 3, stride=1, padding=1),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        self.unpool_block13 = Sequential(
            # unpool3
            MaxUnpool2d(2, stride=2),
        )

        # 24*24

        self.deconv_block14 = Sequential(
            # deconv3_1
            ConvTranspose2d(256, 256, 3, stride=1, padding=1),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv3_2
            ConvTranspose2d(256, 256, 3, stride=1, padding=1),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv3_3
            ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        self.unpool_block15 = Sequential(
            # unpool2
            MaxUnpool2d(2, stride=2),
        )

        # 48*48

        self.deconv_block16 = Sequential(
            # deconv2_1
            ConvTranspose2d(128, 128, 3, stride=1, padding=1),
            BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv2_2
            ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        self.unpool_block17 = Sequential(
            # unpool1
            MaxUnpool2d(2, stride=2),
        )

        # 96*96

        self.deconv_block18 = Sequential(
            # deconv1_1
            ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
            # deconv1_2
            ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

        self.seg_score19 = Sequential(
            # seg-score
            Conv2d(64, self.n_classes, 1),
        )

        if self.learned_billinear:
            raise NotImplementedError

    def forward(self, x):
        size0 = x.size()
        conv1, indices1 = self.conv_block1(x)
        size1 = conv1.size()
        conv2, indices2 = self.conv_block2(conv1)
        size2 = conv2.size()
        conv3, indices3 = self.conv_block3(conv2)
        size3 = conv3.size()
        conv4, indices4 = self.conv_block4(conv3)
        size4 = conv4.size()
        conv5, indices5 = self.conv_block5(conv4)

        conv6 = self.conv_block6(conv5)
        conv7 = self.conv_block7(conv6)
        conv8 = self.deconv_block8(conv7) + conv5
        conv9 = self.unpool(conv8, indices5, output_size=size4)
        conv10 = self.deconv_block10(conv9) + conv4
        conv11 = self.unpool(conv10, indices4, output_size=size3)
        conv12 = self.deconv_block12(conv11) + conv3
        conv13 = self.unpool(conv12, indices3, output_size=size2)
        conv14 = self.deconv_block14(conv13) + conv2
        conv15 = self.unpool(conv14, indices2, output_size=size1)
        conv16 = self.deconv_block16(conv15) + conv1
        conv17 = self.unpool(conv16, indices1, output_size=size0)
        conv18 = self.deconv_block18(conv17)
        out = self.seg_score19(conv18)

        return out

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
        ]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())
        i_layer = 0
        # copy convolutional filters from vgg16
        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0] : ranges[idx][1]], conv_block):
                if isinstance(l1, Conv2d) and isinstance(l2, Conv2d):
                    if i_layer == 0:
                        l2.weight.data = (
                            (
                                l1.weight.data[:, 0, :, :]
                                + l1.weight.data[:, 1, :, :]
                                + l1.weight.data[:, 2, :, :]
                            )
                            / 3.0
                        ).view(l2.weight.size())
                        l2.bias.data = l1.bias.data
                        i_layer = i_layer + 1
                    else:
                        assert l1.weight.size() == l2.weight.size()
                        assert l1.bias.size() == l2.bias.size()
                        l2.weight.data = l1.weight.data
                        l2.bias.data = l1.bias.data
                        i_layer = i_layer + 1


class TorchPatchDeConvNet(ParameterOperator):
    def __init__(
        self, n_classes=4, learned_billinear=False, clip=0.1, class_weights=False
    ):
        super().__init__(name=type(self).__name__)

        self.model = TorchPatchDeConvNetModule(
            n_classes, learned_billinear, clip, class_weights
        )


class TorchPatchDeConvNetSkip(ParameterOperator):
    def __init__(
        self, n_classes=4, learned_billinear=False, clip=0.1, class_weights=False
    ):
        super().__init__(name=type(self).__name__)

        self.model = TorchPatchDeConvNetSkipModule(
            n_classes, learned_billinear, clip, class_weights
        )


class TorchSectionDeConvNet(ParameterOperator):
    def __init__(
        self, n_classes=4, learned_billinear=False, clip=0.1, class_weights=False
    ):
        super().__init__(name=type(self).__name__)

        self.model = TorchSectionDeConvNetModule(
            n_classes, learned_billinear, clip, class_weights
        )


class TorchSectionDeConvNetSkip(ParameterOperator):
    def __init__(
        self, n_classes=4, learned_billinear=False, clip=0.1, class_weights=False
    ):
        super().__init__(name=type(self).__name__)

        self.model = TorchSectionDeConvNetSkipModule(
            n_classes, learned_billinear, clip, class_weights
        )
