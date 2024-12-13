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
from functions import *
import math


# noinspection SpellCheckingInspection
def main():
    print("Running...")

    # init graphing variables
    terrain_colors = [
        (0.0, "lightblue"),
        (0.1, "darkgreen"),
        (0.5, "green"),
        (0.6, "lightyellow"),
        (0.7, "yellow"),
        (1.0, "#5D3A00")
    ]
    custom_cmap = LinearSegmentedColormap.from_list("CustomTerrain", terrain_colors)

    if os.path.exists("models/save.keras"):
        if input("delete model? (y/)").lower() == "y":
            os.remove("models/save.keras")

    with rasterio.open("datasets/japan_dem_wgs84.tif") as src:
        dem_data = src.read(1)

    # print_data(dem_data)

    # # display japan
    # # invert, normalize and spread
    # display_data = normalize(dem_data)
    # display_data = spread(display_data, 9, 1.5)
    # display_data = invert_above_zero(display_data)

    # # display
    # plt.figure(figsize=(10, 6))
    # plt.imshow(display_data, cmap=custom_cmap)
    # plt.colorbar()
    # plt.show()

    dem_data = normalize(dem_data)

    # show saved graph
    if os.path.exists("models/generated_save.txt"):
        saved = np.loadtxt("models/generated_save.txt")
        plt.figure(num="Saved Graph", figsize=(10, 6))
        plt.imshow(saved, cmap=custom_cmap)
        plt.colorbar()
        plt.show()

    if not os.path.exists("models/save.keras"):
        # now for the RNN
        # sequences
        sequence_length = 50
        sequences, next_values = [], []

        flat_dem = dem_data.flatten()
        for i in range(len(flat_dem) - sequence_length):
            sequences.append(flat_dem[i:i + sequence_length])
            next_values.append(flat_dem[i + sequence_length])

        sequences = np.array(sequences)
        next_values = np.array(next_values)

        # reshape for RNN
        sequences = sequences.reshape((sequences.shape[0], sequence_length, 1))
        print(f"Input shape: {sequences.shape}, Output shape: {next_values.shape}\n")

        # build the RNN
        print("building...")
        model = Sequential([
            SimpleRNN(32, activation='relu', input_shape=(sequence_length, 1)),
            Dense(1)
        ])

        # compile
        print("compiling...")
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # train the model
        print("training...")
        model.fit(sequences, next_values, epochs=1, batch_size=64)

        print("saving...")
        model.save("models/save.keras")
    else:
        print("loading model...")
        # noinspection PyUnresolvedReferences
        model = tf.keras.models.load_model("models/save.keras")

    # create grid
    # CHANGEABLE - likeliness in percent
    likeliness = 30
    print(f"points chosen for grid: {int((100000.*likeliness)/dem_data.size)}")
    grid = generate_grid(dem_data, 50*likeliness)
    # print("grid data:")
    # print_data(grid)
    plt.figure(num="extracted grid", figsize=(10, 6))
    plt.imshow(grid, cmap=custom_cmap)
    plt.colorbar()
    plt.show()

    # generating
    print("generating...")
    generated_terrain = fill_blanks(grid, model)

    # reshape to a 2d grid for visualization
    generated_terrain = np.array(generated_terrain)
    generated_terrain_grid = generated_terrain.reshape((100, 100))

    # format
    generated_terrain_grid = normalize(generated_terrain_grid)
    generated_terrain_grid = spread(generated_terrain_grid, 2, 2)
    generated_terrain_grid[generated_terrain_grid < 0.025] = 0

    # print_data(generated_terrain_grid)

    if input("save? (y/n)").lower() == 'y':
        np.savetxt("models/generated_save.txt", generated_terrain_grid, fmt="%.4f")

    # plot generated terrain
    plt.figure(num="Generated", figsize=(10, 6))
    plt.imshow(generated_terrain_grid, cmap=custom_cmap)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
