import matplotlib.pyplot as plt
import numpy as np


def calculate_rgb(idx, iron_length):
    x = 433 * idx / iron_length
    R = 4.18485e-6*x**3 - 0.00532377*x**2 + 2.19321*x - 39.1125
    G = 1.28826e-10*x**5 - 1.64251e-7*x**4 + 6.73208e-5*x**3 - 0.00808127*x**2 + 0.280643*x - 1.61706
    B = 9.48804e-12*x**5 - 1.05015e-8*x**4 + 4.19544e-5*x**3 - 0.0232532*x**2 + 3.24907*x + 30.466
    
    R = max(0, R)
    G = max(0, G)
    B = max(0, B)
    
    return (R / 255, G / 255, B / 255)

def get_palette_list(iron_length):
    color_list = []
    for idx in range(iron_length):
        color_list.append(calculate_rgb(idx, iron_length))
    return color_list

def get_white_hot(color, palette):
    iron_pal = palette
    color = np.array(color).reshape(1, -1)
    distances = np.linalg.norm(iron_pal - color, axis=1)
    min_index = np.argmin(distances)
    return min_index

def plot_palette(iron_length, convert_to_white_hot=False):
    palette = []
    iron_palette = get_palette_list(iron_length)
    for idx in range(iron_length):
        R, G, B = calculate_rgb(idx, iron_length)
        if convert_to_white_hot:
            idx = get_white_hot((R, G, B), iron_palette)
            rgb = (idx, idx, idx)
        else:
            rgb = (R / 255, G / 255, B / 255)
        palette.append(rgb)
    
    fig, ax = plt.subplots(figsize=(10, 2), nrows=1, ncols=iron_length)
    
    for idx, color in enumerate(palette):
        ax[idx].imshow([[color]])
        ax[idx].axis('off')
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

plot_palette(256, convert_to_white_hot=True)