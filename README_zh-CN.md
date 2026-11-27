<div align="center">

# Open Reality View

**基于环视相机的前馈式生成 3D 驾驶可视化。**<br>
任意虚拟相机，实时，忠于相机所见。

**Jiang Anqing**<sup>\*†</sup> · **Hai Yang** · **Wenliang Su** · **Nan Li** · **Shuo Wang** · **Yang Chen** · **Hao Sun**<sup>✉</sup>

Bosch Corporate Research

<img src="docs/static/images/bosch-logo.svg" alt="Bosch" height="30">

<sub><sup>\*</sup> 第一作者 · <sup>†</sup> 项目负责人 · <sup>✉</sup> 通讯作者</sub>

[项目主页](https://openrealityview.github.io/open-reality-view/) ·
[演示](https://openrealityview.github.io/open-reality-view/#demos) ·
论文（即将发布） ·
模型（即将发布） ·
[English](README.md)

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

</div>

<p align="center">
  <a href="https://openrealityview.github.io/open-reality-view/"><img src="docs/static/images/teaser.jpg" alt="Waymo、Argoverse 2、DDAD 和 nuPlan 上带自车的追尾视角" width="100%"></a>
  <br>
  <em>车辆后上方虚拟相机的追尾视角，覆盖 Waymo、Argoverse 2、DDAD 和 nuPlan，白天和夜晚。视频见<a href="https://openrealityview.github.io/open-reality-view/">项目主页</a>。</em>
</p>

## 概览与方法

<p align="center">
  <img src="docs/static/images/architecture.svg" alt="架构：环视相机、DINO 图像骨干网络、4D 基础模型，然后是三个并行的头：4DGS 头预测 4D 高斯，MeanFlow 头从任意虚拟相机生成 Reality View，规划头输出轨迹" width="100%">
</p>

> Tesla 在 FSD v15 中推出了 Reality View：从车辆上方的虚拟相机实时生成周围环境的画面，由驾驶所用的同一个端到端网络渲染。

Open Reality View 用公开数据和开源组件，以开源方式构建同类系统。环视相机视频先由 DINO 图像骨干网络编码，再由 4D 基础模型跨相机、跨时间融合成一个共享的场景表示。三个头并行读取这一表示：4DGS 头预测场景的 4D 高斯，MeanFlow 头从任意虚拟相机生成 Reality View，规划头输出驾驶轨迹。画面和驾驶决策来自同一个模型。

## 数据集

一套数据契约，适配多种相机配置。ORV 在 nuScenes、nuPlan、Waymo、Argoverse 2、DDAD 和 KITTI 上训练和评测。

每个适配器都输出相同的 `SurroundSample`：同步的多路相机，附带内参、相机到自车的外参和自车位姿，采用统一的约定。模型接受任意标定的相机配置，从前视双目到 8 路环视；追尾视角在每个数据集上含义相同；结果按数据集分别报告，并做留一数据集评测，在模型从未见过的相机配置和城市上测试。

| 数据集 | 相机配置 | 作用 | 许可证 |
|---|---|---|---|
| [nuScenes](https://www.nuscenes.org/nuscenes) | 6 路相机 · 12 Hz · 1,000 个场景 · 波士顿、新加坡 | 主力训练集；丰富的 3D 标注用于忠实度评测。 | CC BY-NC-SA 4.0 |
| [nuPlan / NAVSIM](https://www.nuscenes.org/nuplan) | 8 路相机 · 10 Hz · 120 h 传感器数据 · 4 座城市 | 数据规模；在 NAVSIM 上联合渲染与规划，以及点击即达。 | CC BY-NC-SA 4.0 |
| [Waymo Open Perception](https://waymo.com/open/) | 5 路相机 · 10 Hz · 1,150 个场景 · 无后视相机 | 5 个 LiDAR 提供最好的伪真值；缺少后视可检验生成式补全。 | Waymo 非商业许可 |
| [Argoverse 2 Sensor](https://www.argoverse.org/av2.html) | 7 路环视相机 · 20 Hz · 1,000 个场景 · 美国 6 座城市 | 跨城市泛化；前视相机为竖幅。 | CC BY-NC-SA 4.0 |
| [DDAD](https://github.com/TRI-ML/DDAD) | 6 路相机 · 10 Hz · 全局快门 · 美国和日本 | LiDAR 最远 250 m 的远距离稠密深度。 | CC BY-NC-SA 4.0 |
| [KITTI](https://www.cvlibs.net/datasets/kitti/) | 2 路前视彩色相机（双目）· 10 Hz · 64 线 LiDAR · 卡尔斯鲁厄 | 相机最少的配置：追尾视角大部分需要生成；便于与经典深度和新视角合成基准对比。 | CC BY-NC-SA 3.0 |

上述数据集均仅限非商业使用；每个发布的权重都会注明其训练数据的许可证。

## 演示

同一片段的前视与追尾视角，nuScenes、nuPlan、Waymo、Argoverse 2 和 DDAD 每个数据集两段。追尾视角来自车辆后上方的虚拟相机，自车为插入的 3D 模型。

**自由视角**：同一段驾驶，虚拟相机绕车环绕（orbit）或在车两侧之间摆动（swing），覆盖 Argoverse 2 和 Waymo。

全部视频见[项目主页](https://openrealityview.github.io/open-reality-view/#demos)。

## 状态

本仓库目前托管项目主页。代码、预训练模型和论文即将发布。

## 许可证

代码以 [Apache License 2.0](LICENSE) 发布。第三方代码、权重和数据集沿用各自的许可证。

## 免责声明

Open Reality View 是独立的开源研究项目，与 Tesla, Inc. 不存在任何隶属、认可或赞助关系。文中提及 Tesla、FSD 和 Reality View 仅用于说明启发本项目的产品，相关商标归其所有者。本项目是研究软件，不得用于驾驶车辆或任何安全关键系统。
