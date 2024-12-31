import torch
import torch.nn as nn
import pdb
import os
import pickle

class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ResContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)
        resA = self.conv2(shortcut)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)
        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)
        output = shortcut + resA2
        return output


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True, return_skip=True):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.return_skip = return_skip
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)
        resA = shortcut + resA1

        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)
            if self.return_skip:
                return resB, resA
            else:
                return resB
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB


class UpBlock(nn.Module):

    def __init__(self, in_filters, out_filters, dropout_rate=0.2, drop_out=True, mid_filters=None):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.mid_filters = mid_filters if mid_filters else in_filters // 4 + 2 * out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(self.mid_filters, out_filters, (3, 3), padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip):
        upA = nn.PixelShuffle(2)(x)
        if self.drop_out:
            upA = self.dropout1(upA)

        upB = torch.cat((upA, skip), dim=1)
        if self.drop_out:
            upB = self.dropout2(upB)

        upE = self.conv1(upB)
        upE = self.act1(upE)
        upE1 = self.bn1(upE)

        upE = upE1
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE


class UNet2D(nn.Module):
    def __init__(
        self,
        input_dim=3,
        num_class=20,
        ):
        super(UNet2D, self).__init__()

        self.input_dim = input_dim
        self.num_class = num_class
        first_r, r = 1, 1
        self.cr = 1.0
        self.cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]

        self.cs = [int(self.cr * x) for x in self.cs]
        self.stem = nn.Sequential(
            ResContextBlock(input_dim, self.cs[0]),
            ResContextBlock(self.cs[0], self.cs[0]),
            ResContextBlock(self.cs[0], self.cs[0])
        )
        self.stage1 = ResBlock(self.cs[0] * first_r, self.cs[1], 0.2, pooling=True, drop_out=False)
        self.stage2 = ResBlock(self.cs[1], self.cs[2], 0.2, pooling=True)
        self.stage3 = ResBlock(self.cs[2], self.cs[3], 0.2, pooling=True)
        self.stage4 = ResBlock(self.cs[3], self.cs[4], 0.2, pooling=True)

        self.mid_stage = ResBlock(self.cs[4], self.cs[4], 0.2, pooling=False)

        self.up1 = UpBlock(self.cs[4] * r, self.cs[5], 0.2, mid_filters=self.cs[4] * r // 4 + self.cs[4])
        self.up2 = UpBlock(self.cs[5], self.cs[6], 0.2, mid_filters=self.cs[5] // 4 + self.cs[3])
        self.up3 = UpBlock(self.cs[6] * r, self.cs[7], 0.2, mid_filters=self.cs[6] * r // 4 + self.cs[2])
        self.up4 = UpBlock(self.cs[7], self.cs[8], 0.2, drop_out=False, mid_filters=self.cs[7] // 4 + self.cs[1])

        self.classifier = nn.Sequential(
            nn.Conv2d(self.cs[8], self.num_class, kernel_size=1, stride=1)
        )

    def forward(self, data_dict):
        batch_size = len(data_dict['offset_img'])
        x = data_dict['image_input']
        c, h, w = x.shape[1], x.shape[2], x.shape[3]

        x = self.stem(x) # 36, 32, 384, 1280
        x1, s1 = self.stage1(x)  #  36,  32, 192, 640 / 36,  32, 384, 1280
        x2, s2 = self.stage2(x1) #  36,  64,  96, 320 / 36,  64, 192, 640
        x3, s3 = self.stage3(x2) #  36, 128,  48, 160 / 36, 128,  96, 320
        x4, s4 = self.stage4(x3) #  36, 256,  24,  80 / 36, 256,  48, 160

        x5 = self.mid_stage(x4)  #  36, 256, 24, 80

        u1 = self.up1(x5, s4) # 36, 256, 48, 160
        u2 = self.up2(u1, s3) # 36, 128, 96, 320
        u3 = self.up3(u2, s2) # 36, 96, 192, 640
        u4 = self.up4(u3, s1) # 36, 96, 384, 1280

        image_logits = self.classifier(u4)
        data_dict['image_logits'] = image_logits

        img_scale0_logits = []
        img_scale0_targets = []
        img_scale0_rgb = []
        img_scale0 = []
        img_scale4 = []
        for batch_idx in range(batch_size):
            start = data_dict['offset_img'][batch_idx-1] if batch_idx > 0 else 0
            end = data_dict['offset_img'][batch_idx]
            img_scale0_logits.append(data_dict['image_logits'].permute(0, 2, 3, 1)[start:end].reshape(-1, w, self.num_class))
            img_scale0_targets.append(data_dict['semantic_map_ms'].permute(0, 2, 3, 1)[start:end].reshape(-1, w))
            img_scale0_rgb.append(data_dict['image_input'].permute(0, 2, 3, 1)[start:end].reshape(-1, w, c))
            img_scale0.append(u4.permute(0, 2, 3, 1)[start:end].reshape(-1, w, self.cs[8]))
            img_scale4.append(u2.permute(0, 2, 3, 1)[start:end].reshape(-1, w//4, self.cs[6]))

        img_indices = []
        for batch_idx in range(batch_size):
            batch_mask = data_dict['lidar_fov_ms'].C[:, -1] == batch_idx
            img_indices.append(data_dict['lidar_fov_ms'].F[:, -2:][batch_mask])

        img_scale0_logits_ = []
        img_scale0_targets_ = []
        img_scale0_rgb_ = []
        img_scale0_ = []
        img_scale4_ = []
        for i in range(batch_size):
            img_scale0_logits_.append(img_scale0_logits[i][img_indices[i][:, 0].long(), img_indices[i][:, 1].long()])
            img_scale0_targets_.append(img_scale0_targets[i][img_indices[i][:, 0].long(), img_indices[i][:, 1].long()])
            img_scale0_rgb_.append(img_scale0_rgb[i][img_indices[i][:, 0].long(), img_indices[i][:, 1].long()])
            img_scale0_.append(img_scale0[i][img_indices[i][:, 0].long(), img_indices[i][:, 1].long()])
            img_scale4_.append(img_scale4[i][img_indices[i][:, 0].long()//4, img_indices[i][:, 1].long()//4])

        img_scale0_logits_ = torch.cat(img_scale0_logits_, dim=0)
        img_scale0_targets_ = torch.cat(img_scale0_targets_, dim=0)
        img_scale0_rgb_ = torch.cat(img_scale0_rgb_, dim=0)
        img_scale0_ = torch.cat(img_scale0_, dim=0)
        img_scale4_ = torch.cat(img_scale4_, dim=0)
        data_dict['image_logits_fov'] = img_scale0_logits_
        data_dict['image_targets_fov'] = img_scale0_targets_
        data_dict['image_rgb_fov'] = img_scale0_rgb_
        data_dict['image_features_fov'] = torch.cat([img_scale0_, img_scale4_], dim=-1)
            
        return data_dict