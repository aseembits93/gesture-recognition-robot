import numpy as np


def generate_hill(nrow, ncol, debug=False):
    hills   = 1
    noise   = 0.25
    heights = np.random.uniform(0.5, 1, hills)
    locs    = [(np.random.randint(int((1-noise) * nrow/2), int((1+noise) * nrow/2)), np.random.randint(int((1-noise) * ncol/2), int((1+noise) * ncol/2))) for _ in range(hills)]
    dims    = [(np.random.randint(50, 70), np.random.randint(50, 70)) for _ in range(hills)]

    arr = np.zeros((nrow, ncol))
    for hgt, loc, dim in zip(heights, locs, dims):
        x_0, y_0 = loc
        w, l     = dim

        if debug:
            print("generated hill with")
            print("\tloc {:3d} {:3d}".format(x_0, y_0))
            print("\tw {:3d}, l {:3d} -> {:3d}, {:3d}".format(w, l, min(w, (nrow - x_0)//2, x_0//2), min(l, (ncol - y_0)//2, y_0//2)))

        w = min(w, (nrow - x_0)//2, x_0//2)
        l = min(l, (ncol - y_0)//2, y_0//2)

        raw = np.array([hgt * np.sin(q) * np.sin(np.linspace(0, np.pi, num=l)) for q in np.linspace(0, np.pi, num=w)])

        x_start = int(x_0 - (w/2))
        x_end   = int(x_0 + (w/2))
        y_start = int(y_0 - (l/2))
        y_end   = int(y_0 + (l/2))

        if debug:
            print("\tx start {:2d} x end {:2d} y start {:2d} y end {:2d}".format(x_start, x_end, y_start, y_end))
            print("\tarr slice shape", arr[x_start:x_end, y_start:y_end].shape, "vs raw shape", raw.shape)

        arr_slice = (arr[x_start:x_end, y_start:y_end] + raw) / 2
        arr[x_start:x_end, y_start:y_end] = arr_slice
    return arr, (20, 20, 2)

def generate_stairs(nrow, ncol, x_scale=10, y_scale=10, z_scale=1):
    stair_len  = 0.25
    stair_rise = 0.20
    num_stairs = 4

    arr = np.zeros((nrow, ncol))

    x_start = int(nrow // 2 + nrow//2 * 1 / x_scale)

    x_step_span     = int(nrow//2 * stair_len / x_scale)
    stair_rise_span = stair_rise / z_scale

    current_x = x_start

    for step in range(num_stairs):
        next_x = current_x + x_step_span
        print(current_x, next_x, x_step_span, stair_rise_span)
        print(arr[current_x:next_x].shape, 'vs', np.ones((x_step_span, ncol)).shape)
        arr[:,current_x:next_x] = np.ones((ncol, x_step_span)) * stair_rise_span * step
        current_x = next_x
    for step in range(num_stairs)[::-1]:
        next_x = current_x + x_step_span
        print(current_x, next_x, x_step_span, stair_rise_span)
        print(arr[current_x:next_x].shape, 'vs', np.ones((x_step_span, ncol)).shape)
        arr[:,current_x:next_x] = np.ones((ncol, x_step_span)) * stair_rise_span * step
        current_x = next_x
    print(np.max(arr))
    return arr, (x_scale, y_scale, z_scale)

