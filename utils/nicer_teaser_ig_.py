import numpy as np

import constants
from custom_types import *
from utils import files_utils, image_utils
import cv2


class Transition:

    def get_offsets(self, idx, total):
        width = 1920 // scale
        total = total // scale - width
        start, end = self.transition[idx]
        alphas = ease_linear(end - start)
        padding = np.zeros(start)
        alphas = np.concatenate((padding, alphas), axis=0)
        alphas = (alphas * total).astype(np.int32)
        return alphas

    def enter_transition(self, frames: ARRAYS):
        if len(self.frames) != 0:
            raise ValueError
        self.frames.append(len(frames))

    def end_transition(self, frames: ARRAYS):
        if len(self.frames) != 1:
            raise ValueError
        self.frames.append(len(frames))
        self.transition.append((self.frames[0], self.frames[1]))
        self.frames = []

    def __init__(self):
        self.frames = []
        self.transition = []


def easing(ease_func: Callable[[ARRAY], ARRAY]) -> Callable[[int], ARRAY]:

    def ease_(num_frames):
        if num_frames not in cache:
            alpha = np.linspace(0, 1, num_frames)
            alpha = ease_func(alpha)
            alpha[0] = 0
            alpha[-1] = 1
            cache[num_frames] = alpha
        alpha = cache[num_frames]
        return alpha
    cache = {}
    return ease_


@easing
def ease_linear(alpha: ARRAY) -> ARRAY:
    return alpha


@easing
def ease_in_out(alpha: ARRAY) -> ARRAY:
    alpha: ARRAY = (np.cos(np.pi * alpha) - 1) / 2
    return -alpha


@easing
def ease_out(alpha: ARRAY):
    return 1 - (1 - alpha) ** 2


def get_pixel_grid(image: ARRAY) -> ARRAY:
    h, w, _ = image.shape
    ny, nx = np.arange(h), np.arange(w)
    grid = np.meshgrid(nx, ny)
    grid = np.stack((grid[1], grid[0]), axis=2)
    return grid.astype(np.float32)


def get_center(image: ARRAY) -> ARRAY:
    pixels = np.where(image[:, :, -1] == 1)
    center = [float(item.max() + item.min()) / 2 for item in pixels]
    return V(center).astype(np.float32)


def paste(bg: ARRAY, fg: ARRAY, mask: VN = None, in_place: bool = True) -> ARRAY:
    if mask is None:
        mask = fg[:, :, -1] > 0
    if in_place:
        image = bg.copy()
    image[mask] = fg[mask]
    return image


def paste_multi(bg: ARRAY, *fg: ARRAY, in_place: bool = True) -> ARRAY:
    fg = [np.expand_dims(image, axis=0) if image.ndim == 3 else image for image in fg]
    fg_union = np.concatenate(fg, axis=0)
    mask = fg_union[:, :, :, 3] == 0
    fg_union[mask] = -1
    fg = fg_union.max(axis=0)
    fg[fg[:, :, 3] == 0] = 1
    return paste(bg, fg, in_place=in_place)


def circle_enter(bg_a: ARRAY, bg_b: VN, fg: ARRAY, grid: ARRAY, num_steps: int, start_angle=0., cw=False) -> ARRAYS:
    out = []
    center = get_center(fg)
    fg_relevant = fg[:, :, -1] > 0
    vec = grid - center[None, None, :]
    angle = np.arctan2(vec[:, :, 0], vec[:, :, 1])
    if cw:
        angle = -angle + np.pi
    angle = (angle + np.pi + start_angle * np.pi / 180.) % (2 * np.pi)
    # th = np.linspace(2 * np.pi, 0, num_steps + 1)
    th = 2 * np.pi - ease_in_out(num_steps + 1) * 2 * np.pi
    th = th[1:]
    alpha = np.linspace(0, 1, num_steps)
    for i in range(num_steps):
        mask = angle > th[i]
        mask = np.logical_and(mask, fg_relevant)
        if bg_b is None:
            bg = bg_a
        else:
            bg = bg_b * alpha[i] + bg_a * (1 - alpha[i])
        out.append(paste(bg, fg, mask))
        # files_utils.imshow(out[-1])
    return out



def arrow_enter(bg_a: ARRAY, fg: ARRAY, grid: ARRAY, direction: Tuple[float, float], num_steps: int, bg_b: VN = None) -> ARRAYS:
    out = []
    center = get_center(fg)
    fg_relevant = fg[:, :, -1] > 0
    vec = grid - center[None, None, :]
    vals = np.einsum('hwc,c->hw', vec, V(direction))
    vals[~fg_relevant] = vals[fg_relevant].max()
    vals = vals - vals.min()
    vals = vals / vals.max()
    # th = np.linspace(0, 1, num_steps)
    th = ease_in_out(num_steps)
    # alpha = np.linspace(0, 1, num_steps)
    for i in range(num_steps):
        mask = vals <= th[i]
        mask = np.logical_and(mask, fg_relevant)
        if bg_b is None:
            bg = bg_a
        else:
            bg = bg_b * th[i] + bg_a * (1 - th[i])
        out.append(paste(bg, fg, mask))
    return out


def image_int(image_a: ARRAY, image_b: ARRAY, num_steps: int):
    out = []
    alphas = np.linspace(0, 1, num_steps)
    diff = image_b - image_a
    # alpha_map = np.maximum(image_a[:, :, -1], image_b[:, :, -1])
    for alpha in alphas:
        image = image_a + alpha * diff
        # image[:, :, -1] = alpha_map
        out.append(image)
    return out


def load_image(root, name, on_white: bool = False):
    if on_white:
        image = image_utils.rba_to_rgba_path(f'{root}{name}.png')[:H, :W]
    else:
        image = files_utils.load_image(f'{root}{name}.png', 'RGBA')[:H, :W]
        white = image[:, :, :3].mean(-1) > 130
        image[white][:, 3] = 0
    image = cv2.resize(image, dsize=(W // scale, H // scale), interpolation=cv2.INTER_CUBIC)
    image = image
    return image


def enter_frames(frame: ARRAY):
    h, w, c = frame.shape
    width = 1920 // scale
    bg = np.ones((h,  width, 3)) * 255
    bg = np.concatenate((frame, bg.astype(frame.dtype)), axis=1)
    alphas = w - ease_out(100) * w
    out = [bg[:, int(alpha): int(alpha) + width] for alpha in alphas]
    return out


def camera_move(frames: ARRAYS, transition: Transition) -> ARRAYS:
    # first_frames = enter_frames(frames[0])
    # h, w, c = frames
    offsets = transition.get_offsets(0, 3700)
    for i in range(len(frames)):
        frames[i] = frames[i][:, offsets[i]: offsets[i] + 1920 // scale]
    # frames = [frame[:, : 1920 // scale] for frame in frames]
    # frames = first_frames + frames
    return frames


def compress_frames(frames: ARRAYS, new_frames: ARRAYS) -> Tuple[ARRAYS, ARRAY]:
    last_frame = new_frames[-1]
    for i in range(len(new_frames)):
        frames.append(image_utils.rba_to_rgb_arr(new_frames[i]))
    return frames, last_frame


def fade_out(out: ARRAYS, num_frames: int) -> ARRAYS:
    alphas = ease_linear(num_frames + 1)[1:]
    for i in range(num_frames):
        item = out[-num_frames + i].astype(np.float32)
        alpha = alphas[i]
        item = (1 - alpha) * item + alpha * 255.
        out[-num_frames + i] = item.astype(np.uint8)
    return out


def pad_frames(frames: ARRAYS, padding: int) -> ARRAYS:
    for i in range(padding):
        frames.append(frames[-1])
    return frames

def load_all(root, persist=False):
    cache_path = f'{constants.OUT_ROOT}vid/nisest_seq_{scale:02d}.npy'
    if files_utils.is_file(cache_path) and not persist:
        cache = files_utils.load_np(cache_path)
    else:
        images = [load_image(root, f'seq-{i + 1:02d}', True) for i in range(31)]
        marks = [load_image(root, f'marks-{i + 1:02d}') for i in range(10)]
        arrows = [load_image(root, f'arrows-{i + 1:02d}') for i in range(12)]
        edits = [load_image(root, f'edit-{i + 1:02d}') for i in range(3)]
        cache = np.stack(images + marks + arrows + edits, axis=0)
        files_utils.save_np(cache, cache_path)
    cache = cache.astype(np.float16) / 255.
    images = cache[:31]
    marks = cache[31:41]
    arrows = cache[41:53]
    edits = cache[53:56]
    return images, marks, arrows, edits


def paste_stuff(images: ARRAY, marks: ARRAY, arrow: ARRAY, edits: ARRAY) -> ARRAY:
    images[2] = paste_multi(images[2], marks[0])
    images[3] = paste_multi(images[3], marks[:2])
    images[4] = paste_multi(images[4], marks[:3], arrow[:3])
    images[5] = paste_multi(images[5], marks[:3], arrow[:3])
    images[6] = paste_multi(images[6], marks[:4], arrow[:5])
    images[7] = paste_multi(images[7], marks[:4], arrow[:5])
    images[8] = paste_multi(images[8], marks[:4], arrow[:5])
    images[9] = paste_multi(images[9], marks[:4], arrow[:5])
    images[10] = paste_multi(images[10], marks[:4], arrow[:5])
    images[11] = paste_multi(images[11], marks[:5], arrow[:6])
    images[12] = paste_multi(images[12], marks[:5], arrow[:6])
    images[13] = paste_multi(images[13], marks[:5], arrow[:6])
    images[14] = paste_multi(images[14], marks[:5], arrow[:6])
    images[15] = paste_multi(images[15], marks[:5], arrow[:6])
    images[16] = paste_multi(images[16], marks[:6], arrow[:7])
    images[17] = paste_multi(images[17], marks[:6], arrow[:7])
    images[18] = paste_multi(images[18], marks[:7], arrow[:7])
    images[19] = paste_multi(images[19], marks[:8], arrow[:9])
    images[20] = paste_multi(images[20], marks[:8], arrow[:9])
    images[21] = paste_multi(images[21], marks[:9], arrow[:10])
    images[22] = paste_multi(images[22], marks[:9], arrow[:10])
    images[23] = paste_multi(images[23], marks[:9], arrow[:10])
    images[24] = paste_multi(images[24], marks[:9], arrow[:10])
    images[25] = paste_multi(images[25], marks[:9], arrow[:10])
    images[26] = paste_multi(images[26], marks[:], arrow[:])

    images[27] = paste_multi(images[27], marks[:3], arrow[:3])
    images[28] = paste_multi(images[28], marks[:5], arrow[:6])
    images[29] = paste_multi(images[29], marks[:8], arrow[:9])
    images[30] = paste_multi(images[30], marks[:9], arrow[:10])
    return images


# function easeInOutSine(x: number): number {
# return -(cos(PI * x) - 1) / 2;
# }

scale = 1
# W = 7434
# H = 3339

W = 4384
H = 1080





def new_main():
    out = []
    between_frames = 10
    enter_frames = 10
    after_edit_frames = 5
    inter_frames = 10
    root = r'C:\Users\hertz\PycharmProjects\sdf_gmm\assets\illu/nisest_seq/'
    images, marks, arrows, edits = load_all(root, persist=True)
    images = paste_stuff(images, marks, arrows, edits)
    grid = get_pixel_grid(images[0])
    transition = Transition()
    bg = np.ones_like(images[0])
    out, last_frame = compress_frames(out, [bg] * 20)
    out, last_frame = compress_frames(out, image_int(last_frame, images[0], 20))
    out = pad_frames(out, between_frames)
    out, last_frame = compress_frames(out, circle_enter(last_frame, images[1], marks[0], grid, 12, 180., True))
    out, last_frame = compress_frames(out, circle_enter(last_frame, images[2], marks[1], grid, 12, 180., True))
    out, last_frame = compress_frames(out, circle_enter(last_frame, images[3], marks[2], grid, 12, 270., False))
    out, last_frame = compress_frames(out, arrow_enter(last_frame, arrows[0], grid, (0.5, 1), 10))
    out, last_frame = compress_frames(out, arrow_enter(last_frame, arrows[1], grid, (0.2, 1), 10))
    out, last_frame = compress_frames(out, arrow_enter(last_frame, arrows[2], grid, (-1, 1), 10))
    out, last_frame = compress_frames(out, image_int(last_frame, images[4], 20))
    transition.enter_transition(out)
    out = pad_frames(out, between_frames)
    out, last_frame = compress_frames(out, image_int(last_frame, images[27], enter_frames))
    out, last_frame = compress_frames(out, circle_enter(last_frame, images[5], marks[3], grid, 12, 260., False))
    out, last_frame = compress_frames(out, arrow_enter(last_frame, arrows[3], grid, (0., 1), 10))
    out, last_frame = compress_frames(out, arrow_enter(last_frame, arrows[4], grid, (-.3, 1), 10))
    out, last_frame = compress_frames(out, image_int(last_frame, images[6], 20))
    out = pad_frames(out, between_frames)
    out, last_frame = compress_frames(out, image_int(last_frame, images[7], 10))
    out, last_frame = compress_frames(out, arrow_enter(last_frame, edits[0], grid, (.5, 1), 6))
    out = pad_frames(out, after_edit_frames)
    out, last_frame = compress_frames(out, image_int(last_frame, images[8], 20))
    out = pad_frames(out, between_frames)
    out, last_frame = compress_frames(out, image_int(last_frame, images[9], 10))
    out, last_frame = compress_frames(out, circle_enter(last_frame, images[10], marks[4], grid, 12, 185., False))
    out, last_frame = compress_frames(out, arrow_enter(last_frame, arrows[5], grid, (1., 1), 10))
    out, last_frame = compress_frames(out, image_int(last_frame, images[11], 20))
    out = pad_frames(out, between_frames)
    out, last_frame = compress_frames(out, image_int(last_frame, images[12], 10))
    out, last_frame = compress_frames(out, circle_enter(last_frame, None, edits[1], grid, 12, 90., False))
    out = pad_frames(out, after_edit_frames)
    out, last_frame = compress_frames(out, image_int(last_frame, images[13], 20))
    out = pad_frames(out, between_frames)
    out, last_frame = compress_frames(out, image_int(last_frame, images[14], 10))
    out, last_frame = compress_frames(out, image_int(last_frame, images[28], enter_frames))
    out, last_frame = compress_frames(out, circle_enter(last_frame, images[15], marks[5], grid, 12, 170., True))
    out, last_frame = compress_frames(out, arrow_enter(last_frame, arrows[6], grid, (1., -1.), 10))
    out, last_frame = compress_frames(out, image_int(last_frame, images[16], 20))
    out = pad_frames(out, between_frames)
    out, last_frame = compress_frames(out, circle_enter(last_frame, images[17], marks[6], grid, 12, 200., False))
    out, last_frame = compress_frames(out, circle_enter(last_frame, images[18], marks[7], grid, 12, 260., False))
    out, last_frame = compress_frames(out, arrow_enter(last_frame, arrows[7], grid, (-.1, 1.), 10))
    out, last_frame = compress_frames(out, arrow_enter(last_frame, arrows[8], grid, (-.8, 1.), 10))
    out, last_frame = compress_frames(out, image_int(last_frame, images[19], 20))
    out = pad_frames(out, between_frames)
    out, last_frame = compress_frames(out, image_int(last_frame, images[29], enter_frames))
    out, last_frame = compress_frames(out, circle_enter(last_frame, images[20], marks[8], grid, 12, 180., False))
    out, last_frame = compress_frames(out, arrow_enter(last_frame, arrows[9], grid, (-1.2, -1.), 10))
    out, last_frame = compress_frames(out, image_int(last_frame, images[21], 20))
    out = pad_frames(out, between_frames)
    out, last_frame = compress_frames(out, image_int(last_frame, images[22], 10))
    out, last_frame = compress_frames(out, arrow_enter(last_frame, edits[2], grid, (-1, 0.), 6))
    out = pad_frames(out, after_edit_frames)
    out, last_frame = compress_frames(out, image_int(last_frame, images[23], 20))
    out = pad_frames(out, between_frames)
    out, last_frame = compress_frames(out, image_int(last_frame, images[24], 10))
    out, last_frame = compress_frames(out, image_int(last_frame, images[30], enter_frames))
    out, last_frame = compress_frames(out, circle_enter(last_frame, images[25], marks[9], grid, 12, 5., True))
    out, last_frame = compress_frames(out, arrow_enter(last_frame, arrows[10], grid, (.1, 1.), 10))
    out, last_frame = compress_frames(out, arrow_enter(last_frame, arrows[11], grid, (1., 1.), 10))
    out, last_frame = compress_frames(out, image_int(last_frame, images[26], 20))
    out = pad_frames(out, between_frames)
    out = fade_out(out, 90)
    transition.end_transition(out)
    out = camera_move(out, transition)
    image_utils.gif_group(out, root, 28., 'tmp', mp4=True)



def main():
    root = r'C:\Users\hertz\PycharmProjects\sdf_gmm\assets\illu/nicer_seq/'
    image_a = load_image(root, 'nicer-02', True)
    image_b = load_image(root, 'nicer-05', True)
    image_c = load_image(root, 'nicer-06', True)
    image_d = load_image(root, 'nicer-09', True)
    image_e = load_image(root, 'nicer-12', True)
    image_f = load_image(root, 'nicer-13', True)
    image_g = load_image(root, 'nicer-16', True)
    image_h = load_image(root, 'nicer-17', True)
    image_i = load_image(root, 'nicer-18', True)
    image_j = load_image(root, 'nicer-19', True)
    image_k = load_image(root, 'nicer-21', True)
    image_l = load_image(root, 'nicer-26', True)
    image_m = load_image(root, 'nicer-29', True)

    image_n = load_image(root, 'nicer-30', True)
    image_o = load_image(root, 'nicer-31', True)
    image_p = load_image(root, 'nicer-32', True)

    image_q = load_image(root, 'nicer-38', True)
    image_r = load_image(root, 'nicer-39', True)
    image_s = load_image(root, 'nicer-40', True)
    image_t = load_image(root, 'nicer-41', True)
    image_u = load_image(root, 'nicer-43', True)
    image_v = load_image(root, 'nicer-44', True)
    image_w = load_image(root, 'nicer-45', True)
    # image_x = load_image(root, 'nicer-45', True)
    image_x = load_image(root, 'nicer-47', True)
    # image_z = load_image(root, 'nicer-47', True)



    circle_a = load_image(root, 'mark_a')
    circle_b = load_image(root, 'mark_b')
    circle_c = load_image(root, 'mark_c')
    circle_d = load_image(root, 'mark_d')
    circle_e = load_image(root, 'mark_e')
    circle_f = load_image(root, 'mark_f')
    circle_g = load_image(root, 'mark_g')
    circle_h = load_image(root, 'mark_h')
    circle_i = load_image(root, 'mark_i')
    circle_j = load_image(root, 'mark_j')
    circle_k = load_image(root, 'mark_k')
    circle_l = load_image(root, 'mark_l')
    circle_m = load_image(root, 'mark_m')


    arrow_a = load_image(root, 'arrow_a')
    arrow_b = load_image(root, 'arrow_b')
    arrow_c = load_image(root, 'arrow_c')
    arrow_d = load_image(root, 'arrow_d')
    arrow_e = load_image(root, 'arrow_e')
    arrow_f = load_image(root, 'arrow_f')
    arrow_g = load_image(root, 'arrow_g')
    arrow_h = load_image(root, 'arrow_h')
    arrow_i = load_image(root, 'arrow_i')
    arrow_j = load_image(root, 'arrow_j')
    arrow_k = load_image(root, 'arrow_k')
    arrow_l = load_image(root, 'arrow_l')
    arrow_m = load_image(root, 'arrow_m')
    arrow_n = load_image(root, 'arrow_n')
    arrow_o = load_image(root, 'arrow_o')
    arrow_p = load_image(root, 'arrow_p')
    arrow_q = load_image(root, 'arrow_q')



    image_c = paste(image_c, circle_a)
    image_d = paste_multi(image_d, circle_a, circle_b, arrow_a, arrow_b)
    image_e = paste_multi(image_e, circle_a, circle_b, arrow_a, arrow_b)
    image_f = paste_multi(image_f, circle_a, circle_b, arrow_a, arrow_b,circle_c)
    image_g = paste_multi(image_g, circle_a, circle_b, arrow_a, arrow_b, arrow_c, arrow_d, circle_c, circle_d)
    image_h = paste_multi(image_h, circle_a, circle_b, arrow_a, arrow_b, arrow_c, arrow_d, circle_c, circle_d)
    image_i = paste_multi(image_i, circle_a, circle_b, arrow_a, arrow_b, arrow_c, arrow_d, circle_c, circle_d)
    image_j = paste_multi(image_j, circle_a, circle_b, arrow_a, arrow_b, arrow_c, arrow_d, circle_c, circle_d, circle_e)
    image_k = paste_multi(image_k, circle_a, circle_b, arrow_a, arrow_b, arrow_c, arrow_d,
                          arrow_e, arrow_f, circle_c, circle_d, circle_e, circle_f)
    image_l = paste_multi(image_l, circle_a, circle_b, arrow_a, arrow_b, arrow_c, arrow_d,
                          arrow_e, arrow_f, circle_c, circle_d, circle_e, circle_f)
    image_m = paste_multi(image_m, circle_a, circle_b, arrow_a, arrow_b, arrow_c, arrow_d,
                          arrow_e, arrow_f, circle_c, circle_d, circle_e, circle_f, circle_g, arrow_g, arrow_h)

    image_n = paste_multi(image_n, circle_a, circle_b, arrow_a, arrow_b, arrow_c, arrow_d,
                          arrow_e, arrow_f, circle_c, circle_d, circle_e, circle_f, circle_g, arrow_g, arrow_h)

    image_o = paste_multi(image_o, circle_a, circle_b, arrow_a, arrow_b, arrow_c, arrow_d,
                          arrow_e, arrow_f, circle_c, circle_d, circle_e, circle_f, circle_g, arrow_g, arrow_h,
                          circle_h)

    image_p = paste_multi(image_p, circle_a, circle_b, arrow_a, arrow_b, arrow_c, arrow_d,
                          arrow_e, arrow_f, circle_c, circle_d, circle_e, circle_f, circle_g, arrow_g, arrow_h,
                          circle_h, circle_i, arrow_i, arrow_j)

    image_q = paste_multi(image_q, circle_a, circle_b, arrow_a, arrow_b, arrow_c, arrow_d,
                          arrow_e, arrow_f, circle_c, circle_d, circle_e, circle_f, circle_g, arrow_g, arrow_h,
                          circle_h, circle_i, arrow_i, arrow_j)

    image_r = paste_multi(image_r, circle_a, circle_b, arrow_a, arrow_b, arrow_c, arrow_d,
                          arrow_e, arrow_f, circle_c, circle_d, circle_e, circle_f, circle_g, arrow_g, arrow_h,
                          circle_h, circle_i, arrow_i, arrow_j)

    image_s = paste_multi(image_s, circle_a, circle_b, arrow_a, arrow_b, arrow_c, arrow_d,
                          arrow_e, arrow_f, circle_c, circle_d, circle_e, circle_f, circle_g, arrow_g, arrow_h,
                          circle_h, circle_i, arrow_i, arrow_j, circle_j, arrow_k, arrow_l)
    image_t = paste_multi(image_t, circle_a, circle_b, arrow_a, arrow_b, arrow_c, arrow_d,
                          arrow_e, arrow_f, circle_c, circle_d, circle_e, circle_f, circle_g, arrow_g, arrow_h,
                          circle_h, circle_i, arrow_i, arrow_j, circle_j, arrow_k, arrow_l)

    image_u = paste_multi(image_u, circle_a, circle_b, arrow_a, arrow_b, arrow_c, arrow_d,
                          arrow_e, arrow_f, circle_c, circle_d, circle_e, circle_f, circle_g, arrow_g, arrow_h,
                          circle_h, circle_i, arrow_i, arrow_j, circle_j, arrow_k, arrow_l, circle_k)
    image_v = paste_multi(image_v, circle_a, circle_b, arrow_a, arrow_b, arrow_c, arrow_d,
                          arrow_e, arrow_f, circle_c, circle_d, circle_e, circle_f, circle_g, arrow_g, arrow_h,
                          circle_h, circle_i, arrow_i, arrow_j, circle_j, arrow_k, arrow_l, circle_k, circle_l,
                          arrow_m, arrow_n, arrow_o
                          )

    image_w = paste_multi(image_w, circle_a, circle_b, arrow_a, arrow_b, arrow_c, arrow_d,
                          arrow_e, arrow_f, circle_c, circle_d, circle_e, circle_f, circle_g, arrow_g, arrow_h,
                          circle_h, circle_i, arrow_i, arrow_j, circle_j, arrow_k, arrow_l, circle_k, circle_l,
                          arrow_m, arrow_n, arrow_o
                          )

    image_x = paste_multi(image_x, circle_a, circle_b, arrow_a, arrow_b, arrow_c, arrow_d,
                          arrow_e, arrow_f, circle_c, circle_d, circle_e, circle_f, circle_g, arrow_g, arrow_h,
                          circle_h, circle_i, arrow_i, arrow_j, circle_j, arrow_k, arrow_l, circle_k, circle_l,
                          arrow_m, arrow_n, arrow_o, arrow_p, arrow_q, circle_m
                          )

    grid = get_pixel_grid(image_a)
    bg = np.ones_like(image_a)
    out = [bg] * 20
    out += image_int(out[-1], image_a, 20)
    # out = arrow_enter(image_a, arrow_a, grid, (1., 0.), 10)
    out += circle_enter(image_a, image_b, circle_a, grid, 12, 270.)
    out += circle_enter(out[-1], image_c, circle_b, grid, 12, 270.)
    out += arrow_enter(out[-1], arrow_a, grid, (1., 0.), 10)
    out += arrow_enter(out[-1], arrow_b, grid, (0., -1), 10)
    out += image_int(out[-1], image_d, 20)
    out + out[-1] * 10
    out += circle_enter(out[-1], image_e, circle_c, grid, 12, 270.)
    out += circle_enter(out[-1], image_f, circle_d, grid, 12, 270.)
    out += arrow_enter(out[-1], arrow_c, grid, (0., 1), 10)
    out += arrow_enter(out[-1], arrow_d, grid, (1, -1), 10)
    out += image_int(out[-1], image_g, 20)
    out + out[-1] * 10
    out += image_int(out[-1], image_h, 20)
    out += circle_enter(out[-1], image_i, circle_e, grid, 12, 200., cw=True)
    out += circle_enter(out[-1], image_j, circle_f, grid, 12, 200., cw=True)
    out += arrow_enter(out[-1], arrow_e, grid, (-1., 1.5), 10)
    out += arrow_enter(out[-1], arrow_f, grid, (1.2, -1.), 10)
    out += image_int(out[-1], image_k, 20)
    out + out[-1] * 10
    out += circle_enter(out[-1], image_l, circle_g, grid, 12, 80., cw=True)
    out += arrow_enter(out[-1], arrow_g, grid, (1.1, -1.), 10)
    out += arrow_enter(out[-1], arrow_h, grid, (1, 1), 10)
    out += image_int(out[-1], image_m, 20)

    out += circle_enter(out[-1], image_n, circle_h, grid, 12, 70., cw=False)
    out += circle_enter(out[-1], image_o, circle_i, grid, 12, 120., cw=True)

    out += arrow_enter(out[-1], arrow_i, grid, (.1, 1), 10)
    out += arrow_enter(out[-1], arrow_j, grid, (-1., .1), 10)

    out += image_int(out[-1], image_p, 20)
    out += image_int(out[-1], image_q, 20)

    out += circle_enter(out[-1], image_r, circle_j, grid, 12, 60., cw=True)
    out += arrow_enter(out[-1], arrow_k, grid, (.2, -1), 10)
    out += arrow_enter(out[-1], arrow_l, grid, (-.1, 1.), 10)
    out += image_int(out[-1], image_s, 20)

    out += circle_enter(out[-1], image_t, circle_k, grid, 12, 290., cw=True)
    out += circle_enter(out[-1], image_u, circle_l, grid, 12, 265., cw=True)

    out += arrow_enter(out[-1], arrow_m, grid, (.5, 1.), 10)
    out += arrow_enter(out[-1], arrow_n, grid, (-.2, 1.), 10)
    out += arrow_enter(out[-1], arrow_o, grid, (.7, 1.), 10)
    out += image_int(out[-1], image_v, 20)
    out += circle_enter(out[-1], image_w, circle_m, grid, 12, 120., cw=True)
    out += arrow_enter(out[-1], arrow_p, grid, (-1.5, 1.), 10)
    out += arrow_enter(out[-1], arrow_q, grid, (-1., 1.), 10)
    out += image_int(out[-1], image_x, 20)
    out = [image_utils.rba_to_rgb_arr(image) for image in out]
    image_utils.gif_group(out, root, 28.,  'tmp_b', mp4=True)


if __name__ == '__main__':
    new_main()
