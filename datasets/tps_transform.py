import numpy as np
import PIL
from PIL import Image
import torch
import torchvision
from torchvision.transforms import transforms

from scipy import ndimage
from einops import rearrange
from typing import Union
import torch
from torch import nn, Tensor
import random
import matplotlib.pyplot as plt

from torchvision import datasets

# adapted from: https://github.com/eliahuhorwitz/DeepSIM/blob/3cc80fd334c0fc7785296bf70175110df02d4041/data/base_dataset.py#L40
# related blog: https://medium.com/@fanzongshaoxing/image-augmentation-based-on-3d-thin-plate-spline-tps-algorithm-for-ct-data-fa8b1b2a683c


def warp_images(
    from_points, to_points, images, output_region, interpolation_order=1, approximate_grid=10
):
    """Define a thin-plate-spline warping transform that warps from the from_points
    to the to_points, and then warp the given images by that transform. This
    transform is described in the paper: "Principal Warps: Thin-Plate Splines and
    the Decomposition of Deformations" by F.L. Bookstein.
    Parameters:
        - from_points and to_points: Nx2 arrays containing N 2D landmark points.
        - images: list of images to warp with the given warp transform.
        - output_region: the (xmin, ymin, xmax, ymax) region of the output
                image that should be produced. (Note: The region is inclusive, i.e.
                xmin <= x <= xmax)
        - interpolation_order: if 1, then use linear interpolation; if 0 then use
                nearest-neighbor.
        - approximate_grid: defining the warping transform is slow. If approximate_grid
                is greater than 1, then the transform is defined on a grid 'approximate_grid'
                times smaller than the output image region, and then the transform is
                bilinearly interpolated to the larger region. This is fairly accurate
                for values up to 10 or so.
    """
    transform = _make_inverse_warp(from_points, to_points, output_region, approximate_grid)
    return [
        ndimage.map_coordinates(
            np.asarray(image), transform, order=interpolation_order, mode="reflect"
        )
        for image in images
    ]


def _make_inverse_warp(from_points, to_points, output_region, approximate_grid):
    x_min, y_min, x_max, y_max = output_region
    if approximate_grid is None:
        approximate_grid = 1
    x_steps = (x_max - x_min) / approximate_grid
    y_steps = (y_max - y_min) / approximate_grid
    x, y = np.mgrid[x_min : x_max : x_steps * 1j, y_min : y_max : y_steps * 1j]

    # make the reverse transform warping from the to_points to the from_points, because we
    # do image interpolation in this reverse fashion
    transform = _make_warp(to_points, from_points, x, y)

    if approximate_grid != 1:
        # linearly interpolate the zoomed transform grid
        new_x, new_y = np.mgrid[x_min : x_max + 1, y_min : y_max + 1]
        x_fracs, x_indices = np.modf((x_steps - 1) * (new_x - x_min) / float(x_max - x_min))
        y_fracs, y_indices = np.modf((y_steps - 1) * (new_y - y_min) / float(y_max - y_min))
        x_indices = x_indices.astype(int)
        y_indices = y_indices.astype(int)
        x1 = 1 - x_fracs
        y1 = 1 - y_fracs
        ix1 = (x_indices + 1).clip(0, x_steps - 1).astype(int)
        iy1 = (y_indices + 1).clip(0, y_steps - 1).astype(int)
        t00 = transform[0][(x_indices, y_indices)]
        t01 = transform[0][(x_indices, iy1)]
        t10 = transform[0][(ix1, y_indices)]
        t11 = transform[0][(ix1, iy1)]
        transform_x = (
            t00 * x1 * y1 + t01 * x1 * y_fracs + t10 * x_fracs * y1 + t11 * x_fracs * y_fracs
        )
        t00 = transform[1][(x_indices, y_indices)]
        t01 = transform[1][(x_indices, iy1)]
        t10 = transform[1][(ix1, y_indices)]
        t11 = transform[1][(ix1, iy1)]
        transform_y = (
            t00 * x1 * y1 + t01 * x1 * y_fracs + t10 * x_fracs * y1 + t11 * x_fracs * y_fracs
        )
        transform = [transform_x, transform_y]
    return transform


_small = 1e-100


def _U(x):
    return (x**2) * np.where(x < _small, 0, np.log(x))


def _interpoint_distances(points):
    xd = np.subtract.outer(points[:, 0], points[:, 0])
    yd = np.subtract.outer(points[:, 1], points[:, 1])
    return np.sqrt(xd**2 + yd**2)


def _make_L_matrix(points):
    n = len(points)
    K = _U(_interpoint_distances(points))
    P = np.ones((n, 3))
    P[:, 1:] = points
    O = np.zeros((3, 3))
    L = np.asarray(np.bmat([[K, P], [P.transpose(), O]]))
    return L


def _calculate_f(coeffs, points, x, y):
    # a = time.time()
    w = coeffs[:-3]
    a1, ax, ay = coeffs[-3:]
    # The following uses too much RAM:
    distances = _U(
        np.sqrt((points[:, 0] - x[..., np.newaxis]) ** 2 + (points[:, 1] - y[..., np.newaxis]) ** 2)
    )
    summation = (w * distances).sum(axis=-1)
    #    summation = np.zeros(x.shape)
    #    for wi, Pi in zip(w, points):
    #        summation += wi * _U(np.sqrt((x-Pi[0])**2 + (y-Pi[1])**2))
    # print("calc f", time.time()-a)
    return a1 + ax * x + ay * y + summation


def _make_warp(from_points, to_points, x_vals, y_vals):
    from_points, to_points = np.asarray(from_points), np.asarray(to_points)
    err = np.seterr(divide="ignore")
    L = _make_L_matrix(from_points)
    V = np.resize(to_points, (len(to_points) + 3, 2))
    V[-3:, :] = 0
    coeffs = np.dot(np.linalg.pinv(L), V)
    x_warp = _calculate_f(coeffs[:, 0], from_points, x_vals, y_vals)
    y_warp = _calculate_f(coeffs[:, 1], from_points, x_vals, y_vals)
    np.seterr(**err)
    return [x_warp, y_warp]


def _get_regular_grid(image, points_per_dim):
    nrows, ncols = image.shape[0], image.shape[1]
    rows = np.linspace(0, nrows, points_per_dim)
    cols = np.linspace(0, ncols, points_per_dim)
    rows, cols = np.meshgrid(rows, cols)
    return np.dstack([cols.flat, rows.flat])[0]


def _generate_random_vectors(image, src_points, scale):
    dst_pts = src_points + np.random.uniform(-scale, scale, src_points.shape)
    return dst_pts


def _thin_plate_spline_warp(image, src_points, dst_points, keep_corners=True):
    width, height = image.shape[:2]
    if keep_corners:
        corner_points = np.array([[0, 0], [0, width], [height, 0], [height, width]])
        src_points = np.concatenate((src_points, corner_points))
        dst_points = np.concatenate((dst_points, corner_points))
    out = warp_images(
        src_points, dst_points, np.moveaxis(image, 2, 0), (0, 0, width - 1, height - 1)
    )
    return np.moveaxis(np.array(out), 0, 2)


def tps_warp(image, points_per_dim, scale):
    width, height = image.shape[:2]
    src = _get_regular_grid(image, points_per_dim=points_per_dim)
    dst = _generate_random_vectors(image, src, scale=scale * width)
    out = _thin_plate_spline_warp(image, src, dst)
    return out


def tps_warp_2(image, dst, src):
    out = _thin_plate_spline_warp(image, src, dst)
    return out


def __apply_tps(img, tps_params):
    new_im = img
    np_im = np.array(img)
    np_im = tps_warp_2(np_im, tps_params["dst"], tps_params["src"])
    return np_im


def tps_transform(np_im: np.ndarray, return_pytorch: bool = True):
    new_w, new_h, _ = np_im.shape

    src = _get_regular_grid(np_im, points_per_dim=3)
    dst = _generate_random_vectors(np_im, src, scale=0.1 * new_w)
    params = {}
    params["tps"] = {"src": src, "dst": dst}
    new_img = __apply_tps(np_im, params["tps"])
    if return_pytorch:
        new_img = torch.from_numpy(new_img)
    return new_img


class TPSTransform:
    """
    This applies TPS on original img with a prob `p`.
    Return: a Pytorch tensor, shape (c,h,w)
    Note: the orders of PIL image and Pytorch Tensor are differents.
    Example:
        ```
            image = Image.open('a_cat.png')
            image = image.convert("RGB") ## PIL image

            image_array = np.asarray(image)  ## (882, 986, 3)
            image_pytorch = torchvision.transforms.ToTensor()(image)  ## (3, 882, 986)
        ```
    """

    def __init__(self, p=0.5):
        """
        with a probability of `p`, we will apply TPS on the original image
        """
        self.p = p

    def _convert_to_numpy(self, img):
        if isinstance(img, np.ndarray):
            pass  # do nothing
        elif isinstance(img, PIL.Image.Image):
            img = np.asarray(img)
        elif isinstance(img, torch.Tensor):
            img = rearrange(img, "c h w-> h w c")
            img = img.numpy()
        else:
            raise TypeError(f"img type `{type(img)}` not supported")
        return img

    def __call__(self, img: Tensor):
        if random.random() > self.p:
            return img
        else:
            img = self._convert_to_numpy(img)
            img = tps_transform(img, return_pytorch=True)
            img = rearrange(img, "h w c ->c h w")
            return img


# Define the dummy transform function
def dummy_transform(x):
    return x


def _test_tps_transform(
    use_tps: bool = True,
    prob=0.5,
    input_img_name: str = "cat_img/a_cat.png",
    new_img_name: str = "",
):
    """
    Just for testing, with a cute cat in `cat_img` folder.
    prob: probability of applying TPS
    """

    if "chammi" in img_path:
        import skimage

        channel_width = 200
        image = skimage.io.imread(img_path)
        # breakpoint()
        image = np.reshape(image, (image.shape[0], channel_width, -1), order="F")
    else:
        # Read an normal RGB image (e.g., the cat img)
        image = Image.open(input_img_name)
        image = image.convert("RGB")

    if new_img_name == "":
        new_img_name = f'cat_img/{input_img_name.replace(".png", "")}_transformed.png'

    img_size = 224
    no_transform = transforms.Lambda(dummy_transform)
    transform_train = transforms.Compose(
        [
            TPSTransform(p=prob) if use_tps else no_transform,
            transforms.RandomResizedCrop(
                img_size, scale=(1.0, 1.0), ratio=(1.0, 1.0), antialias=True
            ),
        ]
    )

    # image_array = np.asarray(image)
    # print("PIL shape after transforming into numpy", image_array.shape)

    image = torchvision.transforms.ToTensor()(image)
    print("image shape here is", image.shape)
    transformed_img = transform_train(image)
    # plt.imshow(transformed_img.permute(1, 2, 0))
    ## save the image
    torchvision.utils.save_image(transformed_img[:3, ...], new_img_name)
    print("wrote transformed image to", new_img_name)
    return transformed_img


if __name__ == "__main__":
    n_imgs = 10
    prob = 1
    use_tps = True
    # img_path = "cat_img/a_cat.png"
    # new_img_path = "cat_img/a_cat_transformed_{i}.png"
    # final_img = f"cat_img/all_transformed_useTPS{use_tps}_prob{prob}.png"

    img_path = "chammi_sample_img/chammi_pic.png"
    new_img_path = "chammi_sample_img/chammi_transformed_{i}.png"
    final_img = f"chammi_sample_img/all_useTPS{use_tps}_prob{prob}.png"

    for i in range(1, n_imgs + 1):
        transformed_img = _test_tps_transform(use_tps, prob, img_path, new_img_path.format(i=i))

    print(transformed_img)

    # show all images in 1 picture
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    for i in range(1, n_imgs + 1):
        img = Image.open(new_img_path.format(i=i))
        axs[(i - 1) // 5, (i - 1) % 5].imshow(img)
        axs[(i - 1) // 5, (i - 1) % 5].axis("off")
    ## store it
    plt.savefig(final_img)
    print("wrote all transformed images to", final_img)
