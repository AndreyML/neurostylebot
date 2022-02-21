from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.utils import save_image
import torchvision.transforms as transforms
import torchvision.models as models

import copy
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageProcessing():

    def __init__(self, content_img_path, style_img_path):
        self.content_image_name = content_img_path
        self.style_image_name = style_img_path

        self.unloader = transforms.ToPILImage()
        self.device = device

        self.imsize = min(256, min(min(self.get_imsize(self.content_image_name)),
                                   min(self.get_imsize(self.style_image_name))))

        self.loader = transforms.Compose([
            transforms.Resize(self.imsize),
            transforms.CenterCrop(self.imsize),
            transforms.ToTensor(),
        ])

        self.device = device


    def get_imsize(self, img_path):
        image = Image.open(img_path).convert('RGB')
        return image.size


    def image_loader(self, img_path):
        image = Image.open(img_path).convert('RGB')
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)

class ContentLoss(nn.Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()
        self.loss = F.mse_loss(self.target, self.target)  # to initialize with something

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def get_gram_matrix(input):
    batch_size, f_map_num, h, w = input.size()

    features = input.view(batch_size * f_map_num, h * w)
    G = torch.mm(features, features.t())
    return G.div(batch_size * h * w * f_map_num)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = get_gram_matrix(target_feature).detach()
        self.loss = F.mse_loss(self.target, self.target)

    def forward(self, input):
        G = get_gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


def get_input_optimizer(input_img_tensor):
    optimizer = optim.LBFGS([input_img_tensor.requires_grad_()])
    return optimizer


class NST:

    def __init__(self, normalization_mean=None, normalization_std=None, content_layers=None, style_layers=None,
                 cnn=None):

        self.normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device) \
            if normalization_mean is None else normalization_mean

        self.normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device) \
            if normalization_std is None else normalization_std

        self.content_layers = ['conv_4'] \
            if content_layers is None else content_layers

        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'] \
            if style_layers is None else style_layers

        # we use pretrained model to extract features
        self.cnn = models.vgg19().load_state_dict(torch.load("pretrained/myvgg")).eval() if cnn is None else cnn

        self.output = None

    def get_style_model_and_losses(self, style_img_tensor, content_img_tensor):
        cnn = copy.deepcopy(self.cnn)

        normalization = Normalization(self.normalization_mean, self.normalization_std).to(device)

        content_losses = []
        style_losses = []

        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in self.content_layers:
                target = model(content_img_tensor).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                target_feature = model(style_img_tensor).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    def run_style_transfer(self, content_img_tensor, style_img_tensor, input_img_tensor,
                           num_steps=200, style_weight=10000000, content_weight=1):

        """Run the style transfer."""
        import time
        start_time = time.time()
        print('Start training...')

        print('Building the style transfer model..')
        model, style_losses, content_losses = \
            self.get_style_model_and_losses(style_img_tensor, content_img_tensor)
        print('Execution time: %s seconds' % (time.time() - start_time))

        optimizer = get_input_optimizer(input_img_tensor)

        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                input_img_tensor.data.clamp_(0, 1)

                optimizer.zero_grad()

                model(input_img_tensor)

                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print("Execution time: %s seconds" % (time.time() - start_time))
                    print()

                return style_score + content_score

            optimizer.step(closure)

        input_img_tensor.data.clamp_(0, 1)

        return input_img_tensor

    def run(self, style_image_name, content_image_name):
        import time
        start_time = time.time()

        print('Getting pictures...')
        image_proc = ImageProcessing(content_image_name, style_image_name)
        print('Execution time: %s seconds' % (time.time() - start_time))

        # print('Showing pictures')
        # image_proc.show_images()
        # print('Execution time: %s seconds' % (time.time() - start_time))

        # running
        print('Loading content image...')
        content_img = image_proc.image_loader(content_image_name)
        print('Execution time: %s seconds' % (time.time() - start_time))

        input_img = content_img.clone()
        print('Loading style image...')
        style_img = image_proc.image_loader(style_image_name)
        print('Execution time: %s seconds' % (time.time() - start_time))

        output = self.run_style_transfer(content_img, style_img, input_img)
        print('Saving the result...')
        self.output = output
        print('Execution time: %s seconds' % (time.time() - start_time))

    def save_result_as_png(self):

        if self.output is None:
            raise TypeError('Nothing to save')
        else:
            save_image(self.output[0], 'result.jpg')
            print('Result saved as result.png')


async def main(content_img_path, style_img_path):
    start_time = time.time()

    # content_image_name = "drive/MyDrive/NST_proj/images/city-sunset.jpg"  # change path if needed
    # style_image_name = "drive/MyDrive/NST_proj/images/city-day.jpg"
    mymodel = models.vgg19(pretrained=False)
    mymodel.features[:13].load_state_dict(torch.load('pretrained/myvgg'))
    vgg = mymodel.features.to(device).eval()
    nst_model = NST(cnn=vgg)
    nst_model.run(style_img_path, content_img_path)

    save_image(nst_model.output[0], 'images/result/result.jpg')

    return 0
