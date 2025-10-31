import numpy as np
import torch
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import visiontransformer

def nc2npz(nc_filename, npz_filename, parameters=['thetao', 'so', 'uo', 'vo'], layers=[0,2,4,6,8]):
    dataset = xr.open_dataset(nc_filename)
    # lats = dataset['latitude'].values
    # lons = dataset['longitude'].values
    n_start = 1176
    n_end = 1596
    e_start = 3564
    e_end = 3876
    final_dataset = []
    for i in parameters:
        layers_dataset = []
        for j in layers:
            data_array = dataset[i][0][j][n_start:n_end, e_start:e_end]
            # print(data_array.shape)
            # print(np.nanmax(data_array))
            # print(np.nanmin(data_array))
            arr = np.nan_to_num(data_array, nan=0.0, posinf=1e5, neginf=-1e5)
            layers_dataset.append(arr)
        final_dataset.append(layers_dataset)
    np.savez_compressed(npz_filename, data = final_dataset)
def img_comparison(actual_dataset, prediction_dataset, var=0, level=0, title="comparison"):

    longitude = np.linspace(117, 143, prediction_dataset.shape[3])
    latitude = np.linspace(18, 53, prediction_dataset.shape[2])

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 2, figsize=(10, 8), subplot_kw={'projection': proj})

    vmin = np.nanmin([prediction_dataset[var][level].flat, actual_dataset[var][level].flat])
    vmax = np.nanmax([prediction_dataset[var][level].flat, actual_dataset[var][level].flat])
    lon_2d, lat_2d = np.meshgrid(longitude, latitude)
    ax1 = ax[0]
    ax1.set_extent([117, 143, 18, 53], crs=ccrs.PlateCarree())
    ax1.add_feature(cfeature.COASTLINE, linewidth=1.2)
    ax1.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1)
    ax1.add_feature(cfeature.LAND, color='lightgray')
    ax1.add_feature(cfeature.OCEAN, color='white')
    c1 = ax1.pcolormesh(lon_2d, lat_2d, actual_dataset[var][level], 
                        cmap='viridis', vmin=vmin, vmax=vmax, 
                        transform=ccrs.PlateCarree(), shading='auto')
    ax1.set_title("Actual Value", fontsize=14)

    ax2 = ax[1]
    ax2.set_extent([117, 143, 18, 53], crs=ccrs.PlateCarree())
    ax2.add_feature(cfeature.COASTLINE, linewidth=1.2)
    ax2.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1)
    ax2.add_feature(cfeature.LAND, color='lightgray')
    ax2.add_feature(cfeature.OCEAN, color='white')
    c2 = ax2.pcolormesh(lon_2d, lat_2d, prediction_dataset[var][level], 
                        cmap='viridis', vmin=vmin, vmax=vmax, 
                        transform=ccrs.PlateCarree(), shading='auto')
    ax2.set_title("Prediction Value", fontsize=14)
    cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.02])
    cbar = fig.colorbar(c1, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(title, fontsize=12)

    plt.subplots_adjust(wspace=0.05, bottom=0.12)

    plt.savefig(title + '.jpg', dpi=300, bbox_inches='tight')
    plt.show()
    
def img_visualization(filename):
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': proj})
    # longitude: 117 ~ 143
    # latitude: 18 ~ 53
    ax.set_extent([117, 143, 18, 53], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=1.2)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1)
    ax.add_feature(cfeature.LAND, color='lightgray')
    ax.add_feature(cfeature.OCEAN, color='white')

    n_start = 1176
    n_end = 1596
    e_start = 3564
    e_end = 3876

    ds = xr.open_dataset(filename)
    lons = ds['longitude'].values
    lats = ds['latitude'].values
    # thetao so uo vo
    thetao_dataset = ds['thetao'][0][0][n_start:n_end, e_start:e_end]
    print(lons[e_start])
    print(lons[e_end])
    print(lats[n_start])
    print(lats[n_end])
    print(thetao_dataset.shape)
    print(lons.shape)
    print(lats.shape)
    vmin = np.nanmin(thetao_dataset)
    vmax = np.nanmax(thetao_dataset)
    print(thetao_dataset[400:,0:2])
    c = thetao_dataset.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap='viridis',
                    vmin=vmin, vmax=vmax, add_colorbar=False,
                    x='longitude', y='latitude')
    ax.set_title("Sea Salinity", fontsize=14)

    cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.02])
    cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal')
    png_name = "Sea Salinity layer 10.jpg"
    cbar.set_label('Sea Salinity')
    plt.savefig(png_name)
    plt.show()

def npz_load(folder_name='NpzDataset-thetaosouovo', key='data'):
    npz_list = os.listdir(folder_name)
    final_dataset = []
    for i in npz_list:
        dataset = np.load(folder_name + '/' + i)
        print(dataset[key].shape)
        final_dataset.append(dataset[key])
    np.savez_compressed('npz_dataset', data = final_dataset)

if __name__ == '__main__':
    # Step 1 Save to Npz(Monthly)
    # file_list = os.listdir('Dataset')
    # for i in file_list:
    #     print(i.split('.')[0])
    #     nc2npz('Dataset/' + i, 'NpzDataset-thetaosouovo/' + i.split('.')[0])

    # step 2 Combinate Monthly Array
    # npz_load()

    dataset = np.load('npz_dataset.npz')
    print(dataset['data'][0][0][0][400:][0:10])

    # img_visualization('Dataset/mercatorglorys12v1_gl12_mean_202401.nc')