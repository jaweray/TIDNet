import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.utils.utils import strLabelConverter


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()

    def forward(self, x):
        y = torch.clamp(x, 0, 1)
        loss = torch.mean(torch.min(1 - y, y))
        return loss


class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()
        self.criterion = CharbonnierLoss()

    def forward(self, input, target):
        mask = torch.zeros_like(target)
        mask[target < 0.5] = 1
        loss = self.criterion(mask * input, mask * target)
        return loss


class OcrLoss(nn.Module):
    def __init__(self, ocr_model, alphabets):
        super(OcrLoss, self).__init__()
        self.criterion = nn.CTCLoss()
        self.ocr_model = ocr_model
        # self.toGray = torchvision.transforms.Grayscale()
        self.converter = strLabelConverter(alphabets)

    def lcs(self, x, y):
        # find the length of the strings
        m = len(x)
        n = len(y)

        # declaring the array for storing the dp values
        L = [[None] * (n + 1) for i in range(m + 1)]

        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif x[i - 1] == y[j - 1]:
                    L[i][j] = L[i - 1][j - 1] + 1
                else:
                    L[i][j] = max(L[i - 1][j], L[i][j - 1])

        # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
        return L[m][n]

    def forward(self, sr, label_, pos):
        batch_size = sr.size(0)
        # sr = self.toGray(sr)
        sr = torch.mean(sr, dim=1, keepdim=True)

        loss = 0
        sum_len = 0
        sum_correct = 0
        tot = 0
        ymin, ymax, xmin, xmax = pos
        for i in range(batch_size):
            for box_label in label_[i]:
                rect = box_label[1]
                if rect.ymin < ymin[i] or rect.ymax > ymax[i] or rect.xmin < xmin[i] or rect.xmax > xmax[i]:
                    continue
                gt_text = box_label[2].replace(' ', '')
                text, length = self.converter.encode([gt_text])
                if text is None or rect.ymin < 0 or rect.ymax < 0 or rect.xmin < 0 or rect.xmax < 0:
                    continue

                temp_ymin = rect.ymin - ymin[i]
                temp_ymax = rect.ymax - ymin[i]
                temp_xmin = rect.xmin - xmin[i]
                temp_xmax = rect.xmax - xmin[i]
                ocr_area = sr[i, :, temp_ymin:temp_ymax, temp_xmin:temp_xmax].unsqueeze(0)
                if ocr_area.size(3) < 8:
                    continue
                ocr_area = F.interpolate(ocr_area, size=(32, int(ocr_area.size(3) * 32 / ocr_area.size(2) * 160/280)))

                with torch.backends.cudnn.flags(enabled=False):
                    preds = self.ocr_model(ocr_area)
                preds_size = Variable(torch.IntTensor([preds.size(0)]))
                temp_loss = self.criterion(preds, text, preds_size, length)
                if temp_loss == float('inf'):
                    continue
                loss += temp_loss

                # ########## calculate recall of predict ##############
                _, preds = preds.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)
                sum_correct += self.lcs(sim_pred, gt_text)
                sum_len += length.item()

                # # save intermediate image
                # tot += 1
                # with open('1.txt', 'a') as fp:
                #     fp.write('{}\t{}\t\t{}\n'.format(tot, sim_pred, gt_text))
                # img = ocr_area.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255
                # img2 = sr[i, :, rect.ymin:rect.ymax+2, rect.xmin:rect.xmax+2].permute(1, 2, 0).cpu().detach().numpy() * 255
                # cv2.imwrite('./temp/{:0>4d}.jpg'.format(tot), img)
                # cv2.imwrite('./temp/{:0>4d}_c.jpg'.format(tot), img2)

        return loss / batch_size, sum_correct, sum_len