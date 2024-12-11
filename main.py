import tensorflow as tf
from matplotlib.colors import LinearSegmentedColormap
# noinspection PyUnresolvedReferences
from tensorflow.keras import Sequential
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import box

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def invert_above_zero(data):
    return np.where(data > 0, 1 - data, data)

def main():
    print("Running...")

    with rasterio.open("datasets/japan_dem_wgs84.tif") as src:
        dem_data = src.read(1)

    print(f"Minimum value in DEM: {np.min(dem_data)}")
    print(f"Maximum value in DEM: {np.max(dem_data)}")

    # invert, normalize and spread
    for _ in range(9):
        dem_data = normalize(dem_data)
        dem_data = dem_data**1.5
    dem_data = normalize(dem_data)
    dem_data = invert_above_zero(dem_data)

    print(f"Minimum value in DEM: {np.min(dem_data)}")
    print(f"Maximum value in DEM: {np.max(dem_data)}")

    print(f"Dataset size: {dem_data.shape}")

    # display tif custom sense
    terrain_colors = [
        (0.0, "lightblue"),
        (0.1, "darkgreen"),
        (0.5, "green"),
        (0.6, "lightyellow"),
        (0.7, "yellow"),
        (1.0, "#5D3A00")
    ]
    custom_cmap = LinearSegmentedColormap.from_list("CustomTerrain", terrain_colors)
    plt.figure(figsize=(10, 6))
    plt.imshow(dem_data, cmap=custom_cmap)
    plt.colorbar()
    plt.show()

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
    print(f"Input shape: {sequences.shape}, Output shape: {next_values.shape}")

    # build the RNN
    model = Sequential([
        SimpleRNN(64, activation='relu', input_shape=(sequence_length, 1)),
        Dense(1)
    ])

    # compile
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # train the model
    model.fit(sequences, next_values, epochs=20, batch_size=64)

    # starting sequence
    starting_sequence = flat_dem[:sequence_length].reshape((1, sequence_length, 1))

    # gen new terrain
    generated_terrain = []
    current_sequence = starting_sequence

    for _ in range(1000):
        next_value = model.predict(current_sequence)
        generated_terrain.append(next_value[0, 0])

        current_sequence = np.append(current_sequence[:, 1:, :], [[next_value]], axis=1)

    # reshape to a 2d grid for visualization
    generated_terrain = np.array(generated_terrain)
    generated_terrain_grid = generated_terrain.reshape((50, 20))

    # plot generated terrain
    custom_cmap = LinearSegmentedColormap.from_list("CustomTerrain", terrain_colors)
    plt.figure(figsize=(10, 6))
    plt.imshow(generated_terrain_grid, cmap=custom_cmap)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
