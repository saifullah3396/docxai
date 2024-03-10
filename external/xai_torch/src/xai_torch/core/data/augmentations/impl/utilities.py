from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike


def disk(radius, alias_blur=0.1, dtype=np.float32):
    """
    Creats the aliased kernel disk over image.
    """
    import cv2
    import numpy as np

    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def clipped_zoom(self, image: ArrayLike, zoom_factor: float):
    """
    Applies clipped zoom over image.
    """
    import numpy as np
    from scipy.ndimage import zoom as scizoom

    h = image.shape[0]
    w = image.shape[1]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / float(zoom_factor)))
    cw = int(np.ceil(w / float(zoom_factor)))
    top = (h - ch) // 2
    left = (w - cw) // 2
    img = scizoom(
        image[top : top + ch, left : left + cw],
        (self.zoom_factor, self.zoom_factor),
        order=1,
    )
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2
    trim_left = (img.shape[1] - w) // 2

    return img[trim_top : trim_top + h, trim_left : trim_left + w]


# class RandomResizedCropCustom(transforms.RandomResizedCrop):
#     @staticmethod
#     def get_params(img, scale, ratio, region_mask):
#         """Get parameters for ``crop`` for a random sized crop.
#         Args:
#             img (PIL Image or Tensor): Input image.
#             scale (list): range of scale of the origin size cropped
#             ratio (list): range of aspect ratio of the origin aspect ratio cropped
#         Returns:
#             tuple: params (i, j, h, w) to be passed to ``crop`` for a random
#             sized crop.
#         """
#         width, height = F._get_image_size(img)
#         area = height * width

#         log_ratio = torch.log(torch.tensor(ratio))
#         for _ in range(10):
#             target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
#             aspect_ratio = torch.exp(
#                 torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
#             ).item()

#             w = int(round(math.sqrt(target_area * aspect_ratio)))
#             h = int(round(math.sqrt(target_area / aspect_ratio)))

#             if 0 < w <= width and 0 < h <= height:
#                 # mask = region_mask[h//2:height-h//2, w//2:width-w//2]
#                 pixel_list = region_mask.nonzero()
#                 if len(pixel_list) == 0:
#                     i = torch.randint(0, height - h + 1, size=(1,)).item()
#                     j = torch.randint(0, width - w + 1, size=(1,)).item()
#                 else:
#                     p_idx = torch.randint(0, len(pixel_list), size=(1,)).item()
#                     i = pixel_list[p_idx][0] - h // 2
#                     j = pixel_list[p_idx][1] - w // 2
#                     i = int(torch.clip(i, min=0, max=height - h - 1))
#                     j = int(torch.clip(j, min=0, max=width - w - 1))

#                 return i, j, h, w

#         # Fallback to central crop
#         in_ratio = float(width) / float(height)
#         if in_ratio < min(ratio):
#             w = width
#             h = int(round(w / min(ratio)))
#         elif in_ratio > max(ratio):
#             h = height
#             w = int(round(h * max(ratio)))
#         else:  # whole image
#             w = width
#             h = height
#         i = (height - h) // 2
#         j = (width - w) // 2
#         return i, j, h, w

#     def forward(self, img, pixel_list):
#         """
#         Args:
#             img (PIL Image or Tensor): Image to be cropped and resized.
#         Returns:
#             PIL Image or Tensor: Randomly cropped and resized image.
#         """
#         i, j, h, w = self.get_params(img, self.scale, self.ratio, pixel_list)
#         return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
