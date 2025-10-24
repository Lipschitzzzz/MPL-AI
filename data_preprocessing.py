import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr

def nc2npz(nc_filename, npz_filename):
    dataset = xr.open_dataset(nc_filename)
    lats = dataset['latitude'].values
    lons = dataset['longitude'].values
    thetao_dataset = dataset['thetao'][0].values
    so_dataset = dataset['so'][0].values
    uo_dataset = dataset['uo'][0].values
    vo_dataset = dataset['vo'][0].values
    # so uo vo
    # print(lats.shape)
    # print(lons.shape)
    # print(dataset.shape)
    # data_array = np.array(dataset)
    print(thetao_dataset.shape)
    # print(thetao_dataset)
    print(so_dataset.shape)
    print(uo_dataset.shape)
    print(vo_dataset.shape)

    print(np.nanmax(np.array(thetao_dataset)))
    print(np.nanmin(np.array(thetao_dataset)))
    print(np.nanmax(np.array(so_dataset)))
    print(np.nanmax(np.array(uo_dataset)))
    print(np.nanmax(np.array(vo_dataset)))
    
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
    # thedao so uo vo
    thetao_dataset = ds['so'][0][10][n_start:n_end, e_start:e_end]
    print(lons[e_start])
    print(lons[e_end])
    print(lats[n_start])
    print(lats[n_end])
    print(thetao_dataset.shape)
    print(lons.shape)
    print(lats.shape)
    vmin = np.nanmin(thetao_dataset)
    vmax = np.nanmax(thetao_dataset)
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


if __name__ == '__main__':
    # nc2npz("./Dataset/mercatorglorys12v1_gl12_mean_202501.nc", "mercatorglorys12v1_gl12_mean_202501.npz")
    filename = "./Dataset/mercatorglorys12v1_gl12_mean_202406.nc"
    img_visualization(filename)