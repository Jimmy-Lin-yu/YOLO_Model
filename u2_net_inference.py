#u2_net_predictor.py
import cv2
import torch
from u2net import U2NET
from torch.autograd import Variable
import numpy as np
import time

class U2NetProcessor:
    def __init__(self, model_path, use_cuda=True):
        """
        Initialize the U2NetProcessor class.
        
        Parameters:
            model_path (str): Path to the pre-trained U2NET model.
            use_cuda (bool): Whether to use GPU acceleration.
        """
        self.model_path = model_path
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.net = self._load_model()

    def _load_model(self):
        """
        Load the U2NET model.
        
        Returns:
            torch.nn.Module: Loaded U2NET model.
        """
        net = U2NET(3, 1)
        net.load_state_dict(torch.load(self.model_path))
        if self.use_cuda:
            net.cuda()
        net.eval()
        return net

    @staticmethod
    def _norm_pred(d):
        """
        Normalize the prediction.
        
        Parameters:
            d (torch.Tensor): Prediction tensor.
            
        Returns:
            torch.Tensor: Normalized prediction tensor.
        """
        ma = torch.max(d)
        mi = torch.min(d)
        return (d - mi) / (ma - mi)

    def inference(self, input_img):
        """
        Perform inference to generate the segmentation mask.
        
        Parameters:
            input_img (numpy.ndarray): Input image in BGR format.peor
            
        Returns:
            numpy.ndarray: Predicted mask.
        """
        tmp_img = np.zeros((input_img.shape[0], input_img.shape[1], 3))
        input_img = input_img / np.max(input_img)

        tmp_img[:, :, 0] = (input_img[:, :, 2] - 0.406) / 0.225
        tmp_img[:, :, 1] = (input_img[:, :, 1] - 0.456) / 0.224
        tmp_img[:, :, 2] = (input_img[:, :, 0] - 0.485) / 0.229

        tmp_img = tmp_img.transpose((2, 0, 1))
        tmp_img = tmp_img[np.newaxis, :, :, :]
        tmp_img = torch.from_numpy(tmp_img).type(torch.FloatTensor)

        if self.use_cuda:
            tmp_img = Variable(tmp_img.cuda())
        else:
            tmp_img = Variable(tmp_img)

        with torch.no_grad():
            d1, _, _, _, _, _, _ = self.net(tmp_img)

        pred = 1.0 - d1[:, 0, :, :]
        pred = self._norm_pred(pred)

        pred = pred.squeeze().cpu().data.numpy()
        return pred

    @staticmethod
    def apply_mask(image, mask, extract_white=True):
        """
        Apply the mask to the original image to extract regions.
        
        Parameters:
            image (numpy.ndarray): Original image.
            mask (numpy.ndarray): Mask image (grayscale, values from 0 to 255).
            extract_white (bool): If True, extract white regions; else extract black regions.
            
        Returns:
            numpy.ndarray: Image with mask applied.
        """
        if len(mask.shape) == 2:
            mask = mask
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if extract_white:
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        else:
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY_INV)

        masked_image = cv2.bitwise_and(image, image, mask=binary_mask)
        return masked_image

    def process_image_direct(self, img, extract_white=True):
        """
        Process an input image directly to generate and apply a mask.
        
        Parameters:
            img (numpy.ndarray): Input image in BGR format.
            extract_white (bool): If True, extract white regions; else extract black regions.
            
        Returns:
            numpy.ndarray: Image with mask applied.
        """
        img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

        start_time = time.time()
        mask = self.inference(img_resized)
        # print(f"Inference time: {time.time() - start_time:.2f} seconds")

        masked_image = self.apply_mask(img_resized, (mask * 255).astype(np.uint8), extract_white)
        return masked_image
