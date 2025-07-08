#!/usr/bin/env python3
"""
完整的大气动力学计算示例
结合ERA5网格处理和du/dt求解
"""

import numpy as np
import matplotlib.pyplot as plt
from WeatherDataProcessor import ERA5GridProcessor, create_test_era5_data,AtmosphericDynamics



def complete_dynamics_workflow():
    """完整的动力学计算工作流"""

    print("Complete Atmospheric Dynamics Workflow")
    print("=" * 60)

    # 1. 配置参数
    config = {
        'domain': {
            'nx': 120,  # 适中的网格大小
            'ny': 100,
            'nz': 40,
            'dx': 50000,  # 50km分辨率
            'dy': 50000,
            'west_lon': -130.0,
            'east_lon': -60.0,
            'south_lat': 20.0,
            'north_lat': 60.0,
            'ref_lon': -95.0,
            'ref_lat': 40.0
        },
        'projection': {
            'type': 'lambert',
            'ref_lat': 40.0,
            'ref_lon': -95.0,
            'truelat1': 30.0,
            'truelat2': 60.0
        }
    }

    print("Step 1: Setting up ERA5 grid processor...")
    processor = ERA5GridProcessor(config)

    # 2. 创建测试数据
    print("\nStep 2: Creating test ERA5 data...")
    era5_file = "test_era5_dynamics.nc"
    create_test_era5_data(era5_file)

    # 3. 处理ERA5数据到WRF网格
    print("\nStep 3: Processing ERA5 data to WRF grid...")
    grid_data = processor.process_era5_to_wrf_grid(era5_file, "wrf_grid_dynamics.nc")

    # 4. 初始化动力学计算器
    print("\nStep 4: Initializing atmospheric dynamics calculator...")
    dynamics = AtmosphericDynamics(grid_data)

    # 5. 计算du/dt和dv/dt
    print("\nStep 5: Calculating du/dt and dv/dt...")
    dudt, dvdt, components = dynamics.calculate_dudt_dvdt(return_components=True)

    # 6. 分析不同层次的动力学
    print("\nStep 6: Analyzing dynamics at different levels...")

    # 分析几个关键层次
    levels_to_analyze = {
        'surface': -1,  # 地面层
        '850hPa': 30,  # 约850hPa
        '500hPa': 25,  # 约500hPa
        '300hPa': 15,  # 约300hPa
        '200hPa': 10  # 约200hPa
    }

    analyses = {}
    for level_name, level_idx in levels_to_analyze.items():
        print(f"  Analyzing {level_name} (level {level_idx})...")
        analyses[level_name] = dynamics.analyze_dynamics(level_idx)

    # 7. 绘制结果
    print("\nStep 7: Plotting results...")
    plot_comprehensive_dynamics(dynamics, analyses)

    # 8. 保存结果
    print("\nStep 8: Saving results...")
    dynamics.save_dynamics_results("atmospheric_dynamics_output.nc")

    # 9. 生成统计报告
    print("\nStep 9: Generating statistics report...")
    generate_dynamics_statistics(dynamics, dudt, dvdt, components)

    print("\nWorkflow completed successfully!")
    return dynamics, analyses


def plot_comprehensive_dynamics(dynamics, analyses):
    """绘制综合动力学分析结果"""

    # 绘制500hPa层的详细分析
    print("  Plotting 500hPa dynamics analysis...")
    dynamics.plot_dynamics_analysis(analyses['500hPa'],
                                    save_path="dynamics_500hPa.png")

    # 绘制垂直剖面
    print("  Plotting vertical cross-sections...")
    plot_vertical_cross_sections(dynamics, analyses)

    # 绘制平流项分析
    print("  Plotting advection terms...")
    plot_advection_analysis(dynamics, analyses['500hPa'])

    # 绘制力平衡分析
    print("  Plotting force balance analysis...")
    plot_force_balance(dynamics, analyses['500hPa'])


def plot_vertical_cross_sections(dynamics, analyses):
    """绘制垂直剖面"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 提取中心经度的垂直剖面
    lon_center_idx = dynamics.nx // 2
    lat = dynamics.coords['latitude'][:, lon_center_idx]
    eta_levels = dynamics.coords['eta_levels'][:-1]

    # 获取各层的数据
    dudt, dvdt, components = dynamics.calculate_dudt_dvdt(return_components=True)

    # 创建垂直坐标网格
    lat_mesh, eta_mesh = np.meshgrid(lat, eta_levels)

    # 1. du/dt的垂直剖面
    dudt_section = dudt[:, :, lon_center_idx] * 1e5
    im1 = axes[0, 0].contourf(lat_mesh, eta_mesh, dudt_section,
                              levels=20, cmap='RdBu_r')
    axes[0, 0].set_title('du/dt Vertical Cross-Section (×10⁻⁵ m/s²)')
    axes[0, 0].set_xlabel('Latitude')
    axes[0, 0].set_ylabel('Eta Level')
    axes[0, 0].invert_yaxis()
    plt.colorbar(im1, ax=axes[0, 0])

    # 2. dv/dt的垂直剖面
    dvdt_section = dvdt[:, :, lon_center_idx] * 1e5
    im2 = axes[0, 1].contourf(lat_mesh, eta_mesh, dvdt_section,
                              levels=20, cmap='RdBu_r')
    axes[0, 1].set_title('dv/dt Vertical Cross-Section (×10⁻⁵ m/s²)')
    axes[0, 1].set_xlabel('Latitude')
    axes[0, 1].set_ylabel('Eta Level')
    axes[0, 1].invert_yaxis()
    plt.colorbar(im2, ax=axes[0, 1])

    # 3. 科里奥利力的垂直剖面
    coriolis_section = components['coriolis_u'][:, :, lon_center_idx] * 1e5
    im3 = axes[1, 0].contourf(lat_mesh, eta_mesh, coriolis_section,
                              levels=20, cmap='viridis')
    axes[1, 0].set_title('Coriolis Force (fv) Vertical Cross-Section (×10⁻⁵ m/s²)')
    axes[1, 0].set_xlabel('Latitude')
    axes[1, 0].set_ylabel('Eta Level')
    axes[1, 0].invert_yaxis()
    plt.colorbar(im3, ax=axes[1, 0])

    # 4. 压力梯度力的垂直剖面
    pgf_section = components['pgf_u'][:, :, lon_center_idx] * 1e5
    im4 = axes[1, 1].contourf(lat_mesh, eta_mesh, pgf_section,
                              levels=20, cmap='plasma')
    axes[1, 1].set_title('Pressure Gradient Force Vertical Cross-Section (×10⁻⁵ m/s²)')
    axes[1, 1].set_xlabel('Latitude')
    axes[1, 1].set_ylabel('Eta Level')
    axes[1, 1].invert_yaxis()
    plt.colorbar(im4, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig("vertical_cross_sections.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_advection_analysis(dynamics, analysis):
    """绘制平流项分析"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    lon = dynamics.coords['longitude']
    lat = dynamics.coords['latitude']

    # 1. u方向平流
    im1 = axes[0, 0].contourf(lon, lat, analysis['u_advection'] * 1e5,
                              levels=20, cmap='RdBu_r')
    axes[0, 0].set_title('u-momentum Advection (×10⁻⁵ m/s²)')
    axes[0, 0].set_xlabel('Longitude')
    axes[0, 0].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axes[0, 0])

    # 2. v方向平流
    im2 = axes[0, 1].contourf(lon, lat, analysis['v_advection'] * 1e5,
                              levels=20, cmap='RdBu_r')
    axes[0, 1].set_title('v-momentum Advection (×10⁻⁵ m/s²)')
    axes[0, 1].set_xlabel('Longitude')
    axes[0, 1].set_ylabel('Latitude')
    plt.colorbar(im2, ax=axes[0, 1])

    # 3. 平流项的量级比较
    u_adv_mag = np.abs(analysis['u_advection'])
    v_adv_mag = np.abs(analysis['v_advection'])
    total_adv_mag = np.sqrt(u_adv_mag ** 2 + v_adv_mag ** 2)

    im3 = axes[1, 0].contourf(lon, lat, total_adv_mag * 1e5,
                              levels=20, cmap='viridis')
    axes[1, 0].set_title('Total Advection Magnitude (×10⁻⁵ m/s²)')
    axes[1, 0].set_xlabel('Longitude')
    axes[1, 0].set_ylabel('Latitude')
    plt.colorbar(im3, ax=axes[1, 0])

    # 4. 垂直速度
    im4 = axes[1, 1].contourf(lon, lat, analysis['vertical_velocity'] * 100,
                              levels=20, cmap='RdBu_r')
    axes[1, 1].set_title('Vertical Velocity (cm/s)')
    axes[1, 1].set_xlabel('Longitude')
    axes[1, 1].set_ylabel('Latitude')
    plt.colorbar(im4, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig("advection_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_force_balance(dynamics, analysis):
    """绘制力平衡分析"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    lon = dynamics.coords['longitude']
    lat = dynamics.coords['latitude']

    # 各项力的量级
    forces = {
        'Advection': np.sqrt(analysis['u_advection'] ** 2 + analysis['v_advection'] ** 2),
        'Coriolis': np.sqrt(analysis['coriolis_u'] ** 2 + analysis['coriolis_v'] ** 2),
        'Pressure Gradient': np.sqrt(analysis['pgf_u'] ** 2 + analysis['pgf_v'] ** 2),
        'Total du/dt': np.sqrt(analysis['dudt'] ** 2 + analysis['dvdt'] ** 2),
        'Geostrophic Wind': np.sqrt(analysis['geostrophic_u'] ** 2 + analysis['geostrophic_v'] ** 2)
    }

    # 绘制各项力的分布
    for i, (name, force) in enumerate(forces.items()):
        if i < 6:  # 只绘制前6个
            row = i // 3
            col = i % 3

            im = axes[row, col].contourf(lon, lat, force * 1e5,
                                         levels=20, cmap='viridis')
            axes[row, col].set_title(f'{name} Magnitude (×10⁻⁵ m/s²)')
            axes[row, col].set_xlabel('Longitude')
            axes[row, col].set_ylabel('Latitude')
            plt.colorbar(im, ax=axes[row, col])

    # 如果有空余的子图，隐藏它
    if len(forces) < 6:
        axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig("force_balance.png", dpi=300, bbox_inches='tight')
    plt.show()


def generate_dynamics_statistics(dynamics, dudt, dvdt, components):
    """生成动力学统计报告"""

    print("\n" + "=" * 60)
    print("ATMOSPHERIC DYNAMICS STATISTICS REPORT")
    print("=" * 60)

    # 基本统计
    print(f"Grid dimensions: {dynamics.nx} × {dynamics.ny} × {dynamics.nz}")
    print(f"Domain coverage: {dynamics.coords['longitude'].min():.1f}° to {dynamics.coords['longitude'].max():.1f}°E")
    print(f"                {dynamics.coords['latitude'].min():.1f}° to {dynamics.coords['latitude'].max():.1f}°N")

    # du/dt和dv/dt的统计
    print(f"\ndu/dt statistics:")
    print(f"  Mean: {np.nanmean(dudt):.2e} m/s²")
    print(f"  Std:  {np.nanstd(dudt):.2e} m/s²")
    print(f"  Min:  {np.nanmin(dudt):.2e} m/s²")
    print(f"  Max:  {np.nanmax(dudt):.2e} m/s²")

    print(f"\ndv/dt statistics:")
    print(f"  Mean: {np.nanmean(dvdt):.2e} m/s²")
    print(f"  Std:  {np.nanstd(dvdt):.2e} m/s²")
    print(f"  Min:  {np.nanmin(dvdt):.2e} m/s²")
    print(f"  Max:  {np.nanmax(dvdt):.2e} m/s²")

    # 各项力的量级比较
    print(f"\nForce magnitude comparison (RMS values):")

    advection_rms = np.sqrt(np.nanmean(components['u_advection'] ** 2 + components['v_advection'] ** 2))
    coriolis_rms = np.sqrt(np.nanmean(components['coriolis_u'] ** 2 + components['coriolis_v'] ** 2))
    pgf_rms = np.sqrt(np.nanmean(components['pgf_u'] ** 2 + components['pgf_v'] ** 2))

    print(f"  Advection:           {advection_rms:.2e} m/s²")
    print(f"  Coriolis force:      {coriolis_rms:.2e} m/s²")
    print(f"  Pressure gradient:   {pgf_rms:.2e} m/s²")

    # 计算各项的相对重要性
    total_force = advection_rms + coriolis_rms + pgf_rms
    print(f"\nRelative importance:")
    print(f"  Advection:           {advection_rms / total_force * 100:.1f}%")
    print(f"  Coriolis force:      {coriolis_rms / total_force * 100:.1f}%")
    print(f"  Pressure gradient:   {pgf_rms / total_force * 100:.1f}%")

    # 动力学特征
    vorticity = dynamics.calculate_vorticity()
    divergence = dynamics.calculate_divergence()

    print(f"\nDynamic characteristics:")
    print(f"  Vorticity RMS:       {np.sqrt(np.nanmean(vorticity ** 2)):.2e} s⁻¹")
    print(f"  Divergence RMS:      {np.sqrt(np.nanmean(divergence ** 2)):.2e} s⁻¹")
    print(f"  Vertical velocity RMS: {np.sqrt(np.nanmean(components['vertical_velocity'] ** 2)):.2e} m/s")

    # 地转平衡检验
    ug, vg = dynamics.calculate_geostrophic_wind()
    u_actual = dynamics.met_fields['u']
    v_actual = dynamics.met_fields['v']

    u_ageostrophic = u_actual - ug
    v_ageostrophic = v_actual - vg

    print(f"\nGeostrophic balance assessment:")
    print(f"  Geostrophic wind RMS:     {np.sqrt(np.nanmean(ug ** 2 + vg ** 2)):.2f} m/s")
    print(f"  Actual wind RMS:          {np.sqrt(np.nanmean(u_actual ** 2 + v_actual ** 2)):.2f} m/s")
    print(f"  Ageostrophic wind RMS:    {np.sqrt(np.nanmean(u_ageostrophic ** 2 + v_ageostrophic ** 2)):.2f} m/s")

    geostrophic_ratio = np.sqrt(np.nanmean(u_ageostrophic ** 2 + v_ageostrophic ** 2)) / np.sqrt(
        np.nanmean(u_actual ** 2 + v_actual ** 2))
    print(f"  Ageostrophic ratio:       {geostrophic_ratio:.3f}")

    print("=" * 60)


def demonstrate_equation_terms():
    """演示动量方程各项的物理意义"""

    print("\n" + "=" * 60)
    print("MOMENTUM EQUATION TERMS EXPLANATION")
    print("=" * 60)

    print("Complete momentum equations:")
    print("du/dt = -u∂u/∂x - v∂u/∂y - w∂u/∂z + fv - (1/ρ)∂p/∂x + friction")
    print("dv/dt = -u∂v/∂x - v∂v/∂y - w∂v/∂z - fu - (1/ρ)∂p/∂y + friction")
    print()

    terms_explanation = {
        "平流项 (Advection)": {
            "u方程": "-u∂u/∂x - v∂u/∂y - w∂u/∂z",
            "v方程": "-u∂v/∂x - v∂v/∂y - w∂v/∂z",
            "物理意义": "由于风场本身的运动导致的动量变化",
            "特点": "非线性项，在强风区和风场变化大的区域重要"
        },

        "科里奥利力 (Coriolis)": {
            "u方程": "+fv",
            "v方程": "-fu",
            "物理意义": "由于地球自转产生的惯性力",
            "特点": "在中高纬度重要，与风速成正比"
        },

        "压力梯度力 (Pressure Gradient)": {
            "u方程": "-(1/ρ)∂p/∂x",
            "v方程": "-(1/ρ)∂p/∂y",
            "物理意义": "由压力差异产生的驱动力",
            "特点": "气象系统的主要驱动力"
        },

        "摩擦力 (Friction)": {
            "u方程": "Fx",
            "v方程": "Fy",
            "物理意义": "地表摩擦和湍流混合的影响",
            "特点": "主要在边界层内重要"
        }
    }

    for term, info in terms_explanation.items():
        print(f"{term}:")
        print(f"  u方程: {info['u方程']}")
        print(f"  v方程: {info['v方程']}")
        print(f"  物理意义: {info['物理意义']}")
        print(f"  特点: {info['特点']}")
        print()

    print("地转平衡 (Geostrophic Balance):")
    print("在大尺度运动中，科里奥利力与压力梯度力平衡：")
    print("fv = -(1/ρ)∂p/∂x")
    print("-fu = -(1/ρ)∂p/∂y")
    print("导出地转风：")
    print("ug = -(1/f)∂Φ/∂y")
    print("vg = (1/f)∂Φ/∂x")
    print("其中Φ是位势高度")

    print("=" * 60)


if __name__ == "__main__":
    # 演示动量方程
    demonstrate_equation_terms()

    # 运行完整的动力学计算工作流
    try:
        dynamics, analyses = complete_dynamics_workflow()
        print("\nAll calculations completed successfully!")

        # 提供进一步分析的建议
        print("\nSuggestions for further analysis:")
        print("1. Examine the balance between different terms")
        print("2. Look at the ageostrophic components")
        print("3. Study the vertical distribution of forces")
        print("4. Analyze the time evolution (if multiple time steps available)")
        print("5. Compare with observational data")

    except Exception as e:
        print(f"Error in workflow: {e}")
        import traceback

        traceback.print_exc()