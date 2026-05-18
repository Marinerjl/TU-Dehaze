import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math

def gaussian(window_size, sigma):
	gauss = torch.Tensor([exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])
	return gauss/gauss.sum()

def create_window(window_size, channel):
	_1D_window = gaussian(window_size, 1.5).unsqueeze(1)
	_2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
	window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
	return window

def SSIM1(img1, img2):
	(_, channel, _, _) = img1.size()
	window_size = 11
	pad = int(window_size/11)
	window = create_window(window_size, channel).to(img1.device)
	mu1 = F.conv2d(img1, window, padding = pad, groups = channel)
	mu2 = F.conv2d(img2, window, padding = pad, groups = channel)

	mu1_sq = mu1.pow(2)
	mu2_sq = mu2.pow(2)
	mu1_mu2 = mu1*mu2

	sigma1_sq = F.conv2d(img1*img1, window, padding = pad, groups = channel) - mu1_sq
	sigma2_sq = F.conv2d(img2*img2, window, padding = pad, groups = channel) - mu2_sq
	sigma12 = F.conv2d(img1*img2, window, padding = pad, groups = channel) - mu1_mu2

	C1 = 0.01**2
	C2 = 0.03**2

	ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
	return ssim_map.mean()


def SSIM(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
	# Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
	if val_range is None:
		if torch.max(img1) > 128:
			max_val = 255
		else:
			max_val = 1

		if torch.min(img1) < -0.5:
			min_val = -1
		else:
			min_val = 0
		L = max_val - min_val
	else:
		L = val_range

	padd = 0
	(_, channel, height, width) = img1.size()
	if window is None:
		real_size = min(window_size, height, width)
		window = create_window(real_size, channel=channel).to(img1.device)

	mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
	mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

	mu1_sq = mu1.pow(2)
	mu2_sq = mu2.pow(2)
	mu1_mu2 = mu1 * mu2

	sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
	sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
	sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

	C1 = (0.01 * L) ** 2
	C2 = (0.03 * L) ** 2

	v1 = 2.0 * sigma12 + C2
	v2 = sigma1_sq + sigma2_sq + C2
	cs = torch.mean(v1 / v2)  # contrast sensitivity

	ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

	if size_average:
		ret = ssim_map.mean()
	else:
		ret = ssim_map.mean(1).mean(1).mean(1)

	if full:
		return ret, cs
	return ret



def PSNR(img1, img2):
	mse = np.mean( (img1/255. - img2/255.) ** 2 )
	if mse == 0:
		return 100
	PIXEL_MAX = 1
	return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# import torch
# import torchvision.transforms.functional as F
# import torchvision.transforms as transforms
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import cv2
# import numpy as np

# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import cv2
# import numpy as np

def calculate_FADE(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Laplacian filter to obtain high-frequency components
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Calculate the mean and standard deviation of the Laplacian
    mean, std = cv2.meanStdDev(laplacian)

    # Calculate the FADE score
    fade_score = np.power(std, 2) / np.power(mean, 2)

    return fade_score.item()

# def FADE(image_tensor):
# 	# Move the image tensor to CPU if it's on GPU
# 	if image_tensor.is_cuda:
# 		image_tensor = image_tensor.cpu()
#
# 	# Convert the image tensor to numpy array
# 	image_np = image_tensor.detach().numpy().squeeze()
#
# 	# Scale the values to 0-255 range
# 	image_np = image_np * 255.0
#
# 	# Convert the numpy array to PIL Image
# 	image_pil = Image.fromarray(np.uint8(image_np.transpose(1, 2, 0)))
#
# 	# Convert PIL Image to OpenCV BGR image
# 	image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
#
# 	# Calculate FADE score
# 	fade_score = calculate_FADE(image_cv)
#
# 	return fade_score








