from hermit_high import *
name_list = ["u_component_of_wind", "v_component_of_wind",
            "temperature", "specific_humidity", "vertical_velocity",
            "geopotential", "level"]

file_path = "era5_day_2021-01-01.nc"
ds = xr.open_dataset(file_path)
ds_curr = ds.sel(time=time_curr)
du_dt_list = []
u_1000_list = []




for time_index in range(0, 24):  # 最后一个点不能算 du_dt，因为缺 u(t+1)
    ds_curr = ds.sel(time=ds.time.values[time_index])

    z_t = ds_curr["geopotential"].values / 9.80665  # z轴

    # 构建 z 插值函数
    all_funcs = build_all_interp_funcs(z_t, ds_curr, varnames)
    data_1000m = interpolate_at_height(all_funcs, 1000,varnames=name_list)
    grad_1000m = differentiate_at_height(all_funcs, 1000, n=1,varnames=name_list)
    grad_2_1000m = differentiate_at_height(all_funcs, 1000, n=2,varnames=name_list)

    # 更新全局变量（必须在 compute_du_dt 外设置）
    globals().update(dict(
        R = 287.0,
        u_1000=data_1000m["u_component_of_wind"],
        v_1000=data_1000m["v_component_of_wind"],
        w_1000=data_1000m["vertical_velocity"],
        temp_1000=data_1000m["temperature"],
        p_1000=data_1000m["level"],
        sp_1000=data_1000m["specific_humidity"],
        du_dz_1000=grad_1000m["u_component_of_wind"],
        dw_dz_1000=grad_1000m["vertical_velocity"],
        du_ddz_100=grad_2_1000m["u_component_of_wind"],
    ))

    # 调用 du_dt 函数
    du_dt = compute_du_dt(ds_curr)

    du_dt_list.append(du_dt)
    u_1000_list.append(data_1000m["u_component_of_wind"])



du_dt_array = np.stack(du_dt_list)            # shape: (T, H, W)
u_1000_array = np.stack(u_1000_list)          # shape: (T, H, W)
np.savez_compressed("du_dt_and_u1000.npz", du_dt=du_dt_array, u=u_1000_array)



