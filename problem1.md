结论是：现在结果差的主要原因不是“SC-GS 的 clone/split 没实现对”，而是当前只复用了 SC-GS 的 Gaussian 初始化和 density control，训练目标、运动图、输入点云和监督方式仍完全不同。当前运行还有明显的 SDS 过强、轮廓监督过弱和过度 densify。

## 当前结果暴露出的直接证据

从 [metrics.json](</home/newdisk/hch/RT-Pose/graph/persistent_4dgs_sds/metrics.json:785>) 看：

| 指标 | 初始化 | Stage 2 后 |
|---|---:|---:|
| 平均 PSNR | 4.39 dB | 24.03 dB |
| 深度 L1 | 0.172 | 0.037 |
| Alpha IoU | 0.390 | 0.284 |

也就是说：

- 颜色和深度数值拟合明显改善；
- 但人物轮廓覆盖反而恶化约 27%；
- 模型学会了在人物 mask 内产生近似颜色，却没有学到干净、紧凑的人体表面。

这与 [最终 summary](</home/newdisk/hch/RT-Pose/graph/persistent_4dgs_sds/persistent_4dgs_summary.png>) 中的现象一致：人物内部颜色误差不算大，但 Gaussian 大量泄漏到轮廓外，呈现黑色噪点和模糊团块。

更明显的是，SDS audit view 从[初始化结果](</home/newdisk/hch/RT-Pose/graph/persistent_4dgs_sds/zero123_sds/audit_views/after_initialization/view_00_az+000_el+00.png>)到[优化后结果](</home/newdisk/hch/RT-Pose/graph/persistent_4dgs_sds/zero123_sds/audit_views/after_stage2/view_00_az+000_el+00.png>)反而变黑、变乱，说明破坏主要发生在 Stage 2。

## 原因一：SDS 实际强度远大于 RGB-D

虽然调度表面上是 `5 RGB-D : 1 SDS`，但梯度强度并不是 5:1。

根据本次 6000 步 history 的均值：

| 梯度 | RGB-D | SDS |
|---|---:|---:|
| center | 0.806 | 6106.8 |
| scale | 0.0085 | 16.49 |
| SH | 0.00083 | 3.39 |
| opacity | 0.00233 | 6.37 |

SDS 的原始 center 梯度平均约为 RGB-D 的 7500 倍。当前 [`optimize_stage2()`](</home/newdisk/hch/RT-Pose/prototype_persistent_4dgs(raw).py:1269>) 没有 `sds_loss_weight`，只是最后统一做 `max_norm=10` 的裁剪。

结果是：

- 几乎每个 SDS step 都会触发强裁剪；
- SDS step 对 Gaussian 产生接近满幅度的更新；
- “5:1 step 比例”不等于“5:1 有效优化比例”；
- SDS 在随机视角下不断改变 center、scale、opacity，破坏了 RGB-D 建立的表面。

当前 [`stable_zero123_sds.py`](</home/newdisk/hch/RT-Pose/stable_zero123_sds.py:266>) 又使用 `reduction="sum"` 构造 SDS loss，因此 loss 数值和反传强度本身就比较大。

这是当前最优先的问题。

## 原因二：RGB-D loss 可以通过“模糊覆盖”获得较好分数

当前 RGB-D loss 在 [`rgbd_training_objective()`](</home/newdisk/hch/RT-Pose/prototype_persistent_4dgs(raw).py:1177>) 中是：

```python
rgb_l1
+ 0.1 * ssim_loss
+ 0.2 * depth_l1
+ 0.05 * coverage
+ 0.01 * leak
```

其中：

- RGB 和 depth 只在动态 mask 内计算；
- mask 外泄漏的权重只有 `0.01`；
- coverage 权重也只有 `0.05`；
- `leak` 又除以整幅图像的大量背景像素。

所以模型可以采用以下退化策略：

1. 把很多大而半透明的 Gaussian 铺在人附近；
2. 保证 mask 内有近似颜色和深度；
3. 即使大量 alpha 泄漏到人物外，惩罚也很小。

这解释了为什么：

- anchor masked PSNR 达到约 27.4 dB；
- 但 silhouette IoU 只有 0.332；
- 肉眼仍然看到模糊、毛刺、黑色团块。

需要增加平衡的 alpha/silhouette loss，而不是只继续增加训练步数。

## 原因三：density control 几乎只有增长，没有清理

Gaussian 数量变化：

```text
26758 → 62554
```

增加约 134%。第一次 density event：

```text
26758 → 48005
clone 21157
split parent 90
```

也就是第一次事件就选中了约 79% 的初始 Gaussian。

整个训练五次 density-control 一共只 prune 了 3 个 Gaussian，并且低 opacity prune 数量始终为 0。这说明当前行为不是“自适应增加并清除错误点”，而是接近单向增殖。

主要问题是：

- `densify-grad-threshold=2e-4` 对当前梯度尺度太低；
- 第一次 densify 产生 21247 个新点；
- 随后全部 48005 个 Gaussian 又被 opacity reset 到 0.01；
- 后续 prune 基本不起作用；
- 大量 clone/split 点被 SDS 推动后形成噪声和 floaters。

SC-GS 的 density control 能工作，是因为它处于稳定的多视角重建和可学习形变场中；不能只复制相同阈值，而忽略当前梯度尺度已经被 RGB-D、SDS 和不同分辨率改变。

## 原因四：第 25 帧 RGB-D 只是单层可见表面

当前初始化只使用第 25 帧：

```text
26998 输入点
26758 graph-supported Gaussian
```

这本质上是一张单视角 depth sheet：

- 人物背面没有真实几何；
- 遮挡区域没有点；
- 手臂、腿部相互遮挡处存在缺口；
- mask/depth 边缘噪声会直接成为 Gaussian；
- 后续只能依赖 SDS 猜测不可见表面。

而 SC-GS 的真实数据通常使用 SfM/COLMAP 点云，并联合学习控制点和时间形变。它并不是“用某一帧深度初始化后，固定运动图，仅靠 SDS 补全”。

因此这里是 `create_from_pcd` 参数初始化方式相似，但输入点云质量并不等价。

## 原因五：固定 K=4 蒙皮过于平均

当前统计显示：

```text
59178 个 Gaussian 使用 4 owners
owner_weight_max_median = 0.257
```

K=4 时均匀权重是 0.25，现在最大权重中位数只有 0.257，说明大量 Gaussian 几乎是四个 owner 平均控制。

这会造成：

- 手臂、腿部边界被相邻节点一起拉动；
- 关节运动被平均；
- 旋转插值后出现体积收缩和拖影；
- 不同身体部位之间产生粘连。

SC-GS 的节点 radius、置信度和 deformation field 是可学习的；当前 [`bind_points_to_graph()`](</home/newdisk/hch/RT-Pose/prototype_persistent_4dgs(raw).py:413>) 的 MoSca owner 和权重是固定的。即使 RGB-D 或 SDS 发现运动错误，也不能调整控制图，只能扭曲 Gaussian 参数补偿。

## 原因六：删除 tether 后，Gaussian 可以脱离原 owner 区域

`center_param` 会被直接优化，但原有 owner binding 不会随 center 更新而重算。于是 Gaussian 可能：

- canonical center 已经移动到另一个身体部位；
- 仍然由原来的四个节点控制；
- SDS 继续把它向随机视角需要的位置推动；
- 时间变形时出现漂浮和错误轨迹。

SC-GS 能联合更新控制节点和 deformation MLP，并用 ARAP 约束局部运动；当前固定 MoSca graph 没有这种自校正能力。

## 原因七：当前实际仍然是 SH=0

当前文件顶部仍是：

```python
MAX_SH_DEGREE: int = 0
```

见 [SH 配置](</home/newdisk/hch/RT-Pose/prototype_persistent_4dgs(raw).py:45>)。

metrics 也明确记录：

```json
"active_degree_final": 3,
"max_degree": 0,
"num_coefficients_per_channel": 1
```

所以 schedule 虽然显示升到 degree 3，实际 tensor 只有一个 DC 系数。当前没有真正的方向相关外观表达。

这不是黑色噪声的首要原因，但会让随机视角 SDS 更难优化。直接恢复 SH3 也不会解决几何问题；在 SDS 过强时，高阶 SH 反而可能重新产生异常颜色。

## 原因八：未来帧变小还与固定相机有关

最终输出使用：

```text
固定 t=25 相机
焦距乘以 0.90
```

见 [最终渲染相机](</home/newdisk/hch/RT-Pose/prototype_persistent_4dgs(raw).py:2004>)。

与此同时，Gaussian 的相机深度从 t=25 附近的约 `0.37–0.55 m` 增加到 t=48 的 `0.57–0.70 m`。人物在固定相机里自然越来越远、越来越小。

Alpha support 也从：

```text
t=25: 51646
t=48: 21884
```

下降约 58%。所以未来帧“缩小、消失”不完全是 Gaussian 外观问题，还包含 MoSca 外推轨迹和固定相机策略。

## 建议的排查顺序

先做一个几乎纯 RGB-D、无 densify 的消融实验：

```bash
python "/home/newdisk/hch/RT-Pose/prototype_persistent_4dgs(raw).py" \
  --stage2-steps 6000 \
  --rgbd-steps-per-sds 5999 \
  --densify-events 0 \
  --output-dir graph/ablation_rgbd_nodensity
```

它会执行 5999 次 RGB-D 和最后 1 次 SDS。如果结果明显变干净，就能确认破坏主要来自 SDS/density control，而不是 MoSca query 或 rasterizer。

随后建议按这个优先级修改：

1. 新增 `--sds-loss-weight`，根据当前梯度统计先从 `1e-4` 到 `1e-3` 测试。
2. 增加平衡的 alpha/silhouette loss，显著提高 mask 外泄漏惩罚。
3. 将 densify threshold 从 `2e-4` 提高一个数量级开始测试，避免第一次复制 79% 的点。
4. 将 K 从 4 改为 3，并减小 `skinning-kappa`，让 owner 权重更局部。
5. 为 canonical center 增加弱 tether，或者移动一定距离后重新绑定 owner。
6. 在几何和 SDS 稳定后，再真正恢复 `MAX_SH_DEGREE=3`。

因此，当前并不是“SC-GS 方法在你的数据上失效”，而是 SC-GS 的两个 Gaussian 操作被放进了一个单帧 depth seed、固定 MoSca graph、强 SDS、弱轮廓约束的不同优化系统里。当前最明确的首要矛盾是：SDS 梯度过强，同时 loss 又允许模糊 Gaussian 覆盖人物区域来获得较高 PSNR。
