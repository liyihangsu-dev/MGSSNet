# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Circle, RegularPolygon
# from matplotlib.projections.polar import PolarAxes
# from matplotlib.projections import register_projection

# def radar_factory(num_vars, frame='circle'):
#     """创建雷达图投影"""
#     theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

#     class RadarAxes(PolarAxes):
#         name = 'radar'

#         def fill(self, *args, **kwargs):
#             return super(RadarAxes, self).fill(theta, *args, **kwargs)

#         def plot(self, *args, **kwargs):
#             lines = super(RadarAxes, self).plot(theta, *args, **kwargs)
#             for line in lines:
#                 self._close_line(line)

#         def _close_line(self, line):
#             x, y = line.get_data()
#             if x[0] != x[-1]:
#                 x = np.concatenate((x, [x[0]]))
#                 y = np.concatenate((y, [y[0]]))
#                 line.set_data(x, y)

#         def set_varlabels(self, labels):
#             self.set_thetagrids(np.degrees(theta), labels)

#         def _gen_axes_patch(self):
#             # 修复 TypeError: 使用显式关键字参数
#             if frame == 'circle':
#                 return Circle((0.5, 0.5), 0.5)
#             else:
#                 return RegularPolygon((0.5, 0.5), numVertices=num_vars, radius=0.5)

#     register_projection(RadarAxes)
#     return theta

# # --- 1. 数据配置 ---
# # 指标标签
# labels = ['PSNR', 'SSIM', 'LPIPS', 'PSNR', 'SSIM', 'LPIPS', 'PSNR', 'SSIM', 'LPIPS']

# # 原始数据 (根据你的图片手动录入)
# raw_data = {
#     'SplatWeaver': [19.09, 0.607, 0.260, 22.96, 0.784, 0.182, 20.15, 0.552, 0.270],
#     'AnySplat':    [17.63, 0.558, 0.281, 22.28, 0.744, 0.201, 18.94, 0.519, 0.300],
#     'YoNoSplat':   [14.64, 0.466, 0.429, 19.73, 0.656, 0.326, 14.54, 0.273, 0.625],
#     'EcoSplat':    [13.16, 0.403, 0.635, 18.45, 0.572, 0.354, 13.16, 0.245, 0.728]
# }

# # --- 2. 数据归一化 (核心步骤) ---
# # 雷达图需要将不同量级的数据映射到相同的视觉比例 (0到1)
# # PSNR/SSIM 越大越好，LPIPS 越小越好 (需要取反)
# def normalize_data(data_dict):
#     arr = np.array(list(data_dict.values()))
#     mins = arr.min(axis=0)
#     maxs = arr.max(axis=0)
#     norm_dict = {}
#     for name, values in data_dict.items():
#         v = np.array(values)
#         norm_v = (v - mins) / (maxs - mins + 1e-6)
#         # 针对 LPIPS 指标索引 (2, 5, 8) 进行取反处理，使其“越大越好”以便展示
#         for i in [2, 5, 8]:
#             norm_v[i] = 1.0 - norm_v[i]
#         norm_dict[name] = norm_v
#     return norm_dict

# norm_data = normalize_data(raw_data)

# # --- 3. 绘图 ---
# N = len(labels)
# theta = radar_factory(N, frame='polygon')

# fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='radar'))
# colors = ['#FF3131', '#A64DFF', '#39FF14', '#FFD700'] 

# # 绘制每个模型的折线和填充
# for (name, values), color in zip(norm_data.items(), colors):
#     ax.plot(values, color=color, label=name, linewidth=2, marker='o', markersize=5)
#     ax.fill(values, facecolor=color, alpha=0.15)

# # 设置刻度标签
# ax.set_varlabels(labels)

# # 绘制三个彩色背景区域 (对应图中的三个 Dataset)
# # 蓝色区域: DL3DV
# ax.fill_between(np.linspace(0, 2*np.pi/3, 100), 0, 1.1, color='#E6F3FF', alpha=0.5, zorder=0)
# # 黄色区域: RealEstate10K
# ax.fill_between(np.linspace(2*np.pi/3, 4*np.pi/3, 100), 0, 1.1, color='#FFF9E6', alpha=0.5, zorder=0)
# # 绿色区域: Mip-NeRF 360
# ax.fill_between(np.linspace(4*np.pi/3, 2*np.pi, 100), 0, 1.1, color='#F0FFF0', alpha=0.5, zorder=0)

# # 在扇区边缘添加数据集名称
# ax.text(np.pi/3, 1.25, 'DL3DV', color='#5D9BCE', weight='bold', ha='center', size=14)
# ax.text(np.pi, 1.25, 'RealEstate10K', color='#D4AF37', weight='bold', ha='center', size=14)
# ax.text(5*np.pi/3, 1.25, 'Mip-NeRF 360', color='#6B8E23', weight='bold', ha='center', size=14)

# # 隐藏原始极坐标刻度
# ax.set_yticklabels([])
# ax.spines['polar'].set_visible(False)

# # 手动添加数据数值标注 (模拟原图红色的数值)
# for i, val in enumerate(raw_data['SplatWeaver']):
#     angle = theta[i]
#     ax.text(angle, 1.05, f'{val}', color='#FF3131', weight='bold', ha='center', va='center')

# # 图例
# ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False, fontsize=12)

# plt.tight_layout()
# plt.show()
# plt.savefig('radar_chart.png')