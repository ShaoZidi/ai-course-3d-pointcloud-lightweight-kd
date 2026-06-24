# 人工智能大作业说明

本文件夹已整理为一个可提交的课程大作业包，主题为“基于轻量化网络与知识蒸馏的 3D 点云目标检测方法设计与分析”。

项目代码开源地址：https://github.com/ShaoZidi/ai-course-3d-pointcloud-lightweight-kd

## 主要交付文件

- report/course_report_submit_v3.pdf：建议提交的课程报告 PDF。
- report/course_report.md：报告 Markdown 源文件，方便继续修改。
- report/course_report.tex：LaTeX 源文件。
- src/：NumPy 实现实验代码。
- tests/：基础单元测试。
- outputs/metrics.json：实验指标原始记录。
- outputs/results_table.md：实验结果表格。

## 重新生成实验结果

使用本地捆绑 Python 或系统 Python 均可，只需要 NumPy 和 ReportLab。

1. 运行实验：

   python src/run_experiment.py

2. 重新生成报告：

   python report/build_report_pdf.py

3. 运行测试：

   python -m unittest discover -s tests -v

## 说明

原论文完整实验依赖 KITTI/nuScenes、OpenPCDet、spconv 和多 GPU。本作业没有伪造这些大规模训练结果，而是受论文启发，采用小规模合成 3D 体素任务完成方案验证：基线模型、轻量化模型和知识蒸馏提升。
