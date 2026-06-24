# 3D 点云轻量检测课程报告设计

## 背景与边界

作业要求是选择一个人工智能相关任务，完成基线实现和方法优化，并按照学术论文结构写课程报告。当前文件夹只有课程要求截图和论文《A Lightweight Model for 3D Point Cloud Object Detection》。论文真实实验依赖 KITTI/nuScenes、OpenPCDet、SECOND、spconv 和多 GPU，当前环境没有数据集和深度学习框架，因此本作业采用“受论文启发的小规模方案验证 + 论文公开结果对照”的方式完成。

报告中会明确说明实验边界：不伪造 KITTI/nuScenes 训练结果；本地实现用于验证轻量化与知识蒸馏思想是否合理，论文表格用于说明原方法在标准数据集上的效果。

## 实验任务

构造一个小规模合成 3D 点云/体素任务。每个样本包含一个简化目标，类别包括 car、pedestrian、cyclist。目标被体素化为固定网格，并加入少量噪声点。模型输出两部分：目标类别和归一化 3D 包围框参数。评价指标包括分类准确率、包围框 MAE、粗略 IoU，以及参数量/乘加量。

## 方法设计

1. 基线模型：使用标准全连接体素网络作为 SECOND 的课程级简化替代。它直接处理展平后的 3D 体素特征，参数量较大，作为 teacher/baseline。
2. 轻量模型：参考论文 LW-Sconv 的思想，用分组线性变换、瓶颈通道和通道混合替代普通全连接层，模拟 group convolution、factorized convolution 和 channel shuffle 的压缩效果。
3. 知识蒸馏：参考论文三部分蒸馏思想，使用 teacher 的 soft label、隐藏特征和关系矩阵约束 student。实现中分别对应 output distillation、feature distillation 和 relation/FSP distillation。

## 交付物

- src/：NumPy 实验代码。
- tests/：基础测试，保证数据生成、统计函数和蒸馏损失可用。
- outputs/：实验指标 JSON 和结果表格。
- report/：中文课程报告 Markdown、LaTeX 源文件和 PDF。

## 验证方式

运行单元测试验证基础模块；运行实验脚本生成指标；报告中引用本地生成的指标与论文公开指标，避免无依据编造。
