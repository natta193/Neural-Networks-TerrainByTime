import tensorflow as tf
from matplotlib.colors import LinearSegmentedColormap
# noinspection PyUnresolvedReferences
from tensorflow.keras import Sequential
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import os
from tqdm import tqdm
import random


def normalize(data):
    # print(data)
    data = data.astype(np.float64)
    # print(data)
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def invert_above_zero(data):
    return np.where(data > 0, 1 - data, data)


def print_data(data):
    print(f"Minimum value in DEM: {np.min(data)}")
    print(f"Maximum value in DEM: {np.max(data)}")
    print(f"Dataset size: {data.shape}")


def spread(data, cycles, factor):
    for _ in range(cycles):
        data = np.sign(data) * (np.abs(data)) ** factor
        data = normalize(data)
    return data


def generate_grid(data: np.typing.NDArray, k) -> np.typing.NDArray: # WORKS
    target_shape = (100, 100)
    rows, cols = data.shape
    new_rows, new_cols = target_shape

    # sf
    row_scale = rows / new_rows
    col_scale = cols / new_cols

    grid = np.zeros(target_shape)

    random_plot_list = random.sample(range(0, data.size), k)

    for point in random_plot_list:
        orig_row, orig_col = divmod(point, cols)
        new_row = int(orig_row / row_scale)
        new_col = int(orig_col / col_scale)

        grid[new_row, new_col] = data[orig_row, orig_col]

    # format
    grid = spread(grid, 3, 1.5)
    grid = invert_above_zero(grid)

    # print_data(grid)
    # print(grid)

    return grid


# noinspection SpellCheckingInspection,PyBroadException
def neighbours(cell_index, row_index, grid):
    # left right up down
    try: # noinspection PyUnboundLocalVariable
        left = grid[row_index][cell_index-1]
    except Exception:
        left = -2.
    try: # noinspection PyUnboundLocalVariable
        right = grid[row_index][cell_index+1]
    except Exception:
        right = -2.
    try: # noinspection PyUnboundLocalVariable
        up = grid[row_index-1][cell_index]
    except Exception:
        up = -2.
    try: # noinspection PyUnboundLocalVariable
        down = grid[row_index+1][cell_index]
    except Exception:
        down = -2.

    neighbourers = [left, right, down, up]

    for neighbour in neighbourers:
        if neighbour == -2.:
            neighbourers.remove(neighbour)

    return neighbourers


# noinspection PyBroadException
def fill_blanks(grid, model):
    grid[grid == 0.] = -1.

    with tqdm(total=grid.size, desc="Filled Cells", unit="cell") as pbar:
        while (grid == -1.).sum() != 0:
            count_neg_ones = (grid == -1.).sum()
            pbar.n = grid.size - count_neg_ones + 1
            pbar.refresh()

            rows, columns = grid.shape

            # predict next
            for row_index in range(rows):
                # load row
                row = grid[row_index, :]

                # load cells in row in random order
                random_cells = np.arange(row.size)
                np.random.shuffle(random_cells)

                # go through cells in row
                for cell_index in random_cells:
                    cell = row[cell_index]
                    if cell != -1.:
                        current_sequence = neighbours(cell_index, row_index, grid)

                        if -1. in current_sequence:
                            current_sequence = [x for x in current_sequence if x != -1.]
                            current_sequence.append(cell)
                            current_sequence = np.array(current_sequence).reshape(1, len(current_sequence), 1)

                            ## MAYBE EXPAND TO SEE FURTHER NEIGHBOURS or change order of sequence?? OR only fill in one neighbour at random ##

                            prediction = model.predict(current_sequence, verbose=0)

                            # # print
                            # print(f"current sequence: {current_sequence}")
                            # print(f"predicted: {prediction}")

                            # fill in -1s
                            # left
                            try:
                                if grid[row_index][cell_index-1] == -1.:
                                    grid[row_index][cell_index-1] = prediction
                            except Exception:
                                pass

                            # right
                            try:
                                if grid[row_index][cell_index+1] == -1.:
                                    grid[row_index][cell_index+1] = prediction
                            except Exception:
                                pass

                            # up
                            try:
                                if grid[row_index-1][cell_index] == -1.:
                                    grid[row_index-1][cell_index] = prediction
                            except Exception:
                                pass

                            # down
                            try:
                                if grid[row_index+1][cell_index] == -1.:
                                    grid[row_index+1][cell_index] = prediction
                            except Exception:
                                pass

                            break

    return grid
