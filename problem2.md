结论：这些“杂色点”主要不是 25→26 帧位姿突然出错，而是 SDS 把随机新视角先验写进了高阶 SH 颜色和 opacity；25 帧后的未来区间没有图像监督，人物运动与遮挡变化把这些颜色异常逐渐暴露出来。

主要原因有四个：

1. SDS 只监督第 25 帧的随机观察方向

训练中的 RGB-D 只从 0–25 帧采样；每隔 6 步添加一次 SDS 时，几何始终查询 anchor=25，然后换随机轨道相机。[training.py](/home/newdisk/hch/RT-Pose/lib_persistent_4dgs/training.py:287)

所以 26–48 帧完全没有 RGB、深度或 SDS loss，只是把训练好的 canonical Gaussians 通过固定 MoSca 轨迹外推。未来帧出现的错误无法再被纠正。

2. SDS 很容易通过高阶 SH“作弊”

当前优化器同时更新位置、scale、opacity、蒙皮参数和 SH：

- `center_param` 学习率：`1e-4`
- `sh_coeffs` 学习率：`2.5e-3`
- `opacity_logits` 学习率：`0.05`

也就是说 SH 学习率是位置的 25 倍。[optimizer.py](/home/newdisk/hch/RT-Pose/lib_persistent_4dgs/optimizer.py:67)

训练到 1500 步又启用了 degree-3 SH，共 16 个颜色系数/通道。[constants.py](/home/newdisk/hch/RT-Pose/lib_persistent_4dgs/constants.py:14) SDS 是随机、低分辨率的生成先验，没有确定的逐像素颜色目标，也没有 SH 正则或颜色范围约束，因此最容易把随机梯度编码成红绿蓝方向性颜色。

我对本次 checkpoint 实算得到：

- 高阶 SH 绝对值 P95：`0.349`
- 最大值：`1.658`
- 在 6 个 SDS 审计视角下，`54%–76%` 的高斯至少一个颜色通道越出 `[0,1]`

渲染器这里只做了 `clamp_min(..., 0)`，没有限制上界：[gauspl_renderer_native_add3.py](/home/newdisk/hch/RT-Pose/lib_render/gauspl_renderer_native_add3.py:155)。最终保存图像再截断到 `[0,1]` 时，不同通道分别饱和，就表现成红、绿、蓝颗粒。

固定第 25 帧几何的 SDS 审计图已经能看到这种现象：[正面审计图](/home/newdisk/hch/RT-Pose/graph/persistent_4dgs_sds/zero123_sds/audit_views/after_stage2/view_00_az+000_el+00.png)、[背面审计图](/home/newdisk/hch/RT-Pose/graph/persistent_4dgs_sds/zero123_sds/audit_views/after_stage2/view_03_az+180_el+15.png)。说明杂色在进入未来运动之前就已经写入图集了。

3. `1e-3` 的 SDS 权重实际上并不小

本次运行共 3000 步，其中 500 步包含 SDS。[metrics.json](/home/newdisk/hch/RT-Pose/graph/persistent_4dgs_sds/metrics.json:893)

虽然配置是 `--sds-loss-weight 1e-3`，但 SDS loss 使用 latent 上的 `reduction="sum"`：[stable_zero123_sds.py](/home/newdisk/hch/RT-Pose/stable_zero123_sds.py:317)。在这 500 个联合更新中：

- 平均加权 SDS loss：约 `0.089`
- 平均 RGB-D reconstruction loss：约 `0.111`
- 最大加权 SDS loss：约 `0.401`

所以 SDS 对部分更新与 RGB-D 是同一量级，不是很弱的辅助项。

4. 密度增长和未来蒙皮进一步放大颗粒

本次高斯数量从 `26,758` 增长到 `89,238`。[metrics.json](/home/newdisk/hch/RT-Pose/graph/persistent_4dgs_sds/metrics.json:809) 大部分新增点都是 clone；第 3000 步还新建了 `5,815` 个 clone，随后训练立即结束，没有后续步骤重新收敛。[metrics.json](/home/newdisk/hch/RT-Pose/graph/persistent_4dgs_sds/metrics.json:1159)

同时：

- SDS 分支没有显式使用 foreground mask；
- RGB-D 背景权重只有 `0.02`；
- 训练后前景外 alpha 泄漏均值从 `0.027` 增加到 `0.448`。[metrics.json](/home/newdisk/hch/RT-Pose/graph/persistent_4dgs_sds/metrics.json:126745)

这些高 opacity、小尺寸、颜色异常的高斯不会被当前 prune 规则删除。未来帧再经过软蒙皮外推，遮挡改变后就会以独立彩色小点的形式露出来。25→26 的全部边界检测实际上都通过了，[metrics.json](/home/newdisk/hch/RT-Pose/graph/persistent_4dgs_sds/metrics.json:127510)，所以它是逐渐累积显现，不是第 26 帧突然跳变。

最直接的验证顺序是：

- 先保持其他参数不变，用 `--sds-loss-weight 0` 做一次消融；
- 再把权重降到 `1e-4 ~ 2e-4`；
- 将 SH 暂时限制为 degree 0 或 1；如果杂色立即消失，就能确定高阶 SH 是直接来源；
- 后续正式修复应考虑：SDS 步冻结高阶 SH、增加 SH/range 正则、避免最后一步 densify，并加强 alpha/leak 约束。

所以一句话概括：SDS 没有直接生成这些彩点，而是其随机梯度在“高学习率、无约束的 degree-3 SH + 大量 densification”中被编码成了颜色异常；26 帧后没有真实监督，未来运动把异常充分显露出来。
