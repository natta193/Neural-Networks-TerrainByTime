import tensorflow as tf
# noinspection PyUnresolvedReferences
from tensorflow.keras import Sequential
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import box

def main():
    print("Running...")

    with rasterio.open("datasets/japan_dem_wgs84.tif") as src:
        dem_data = src.read(1)

    print(f"Dataset size: {dem_data.shape}")

    # first display .tif
    # create a geopandas GeoDataFrame for visualization
    bounds = src.bounds
    geometry = [box(bounds.left, bounds.bottom, bounds.right, bounds.top)]
    gdf = gpd.GeoDataFrame({'geometry':geometry}, crs=src.crs)

    # plot the DEM using geopandas and matplotlib
    fig, ax = plt.subplots(figsize=(8, 8))
    gdf.boundary.plot(ax=ax, color='red', linewidth=2)
    ax.imshow(dem_data, cmap='terrain', extent=src.bounds, origin='upper')
    ax.set_title('Digital Elevation Model')
    plt.show()

    # now for the RNN
    # normalize
    dem_data = (dem_data - np.min(dem_data)) / (np.max(dem_data) - np.min(dem_data))

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
    model.fit(sequences, next_values, epochs=5, batch_size=64)

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

    # plot the generated terrain
    plt.imshow(generated_terrain_grid, cmap='terrain')
    plt.colorbar()
    plt.title("Generated Terrain")
    plt.show()


if __name__ == "__main__":
    main()
