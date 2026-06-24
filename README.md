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


2. 运行测试：

   python -m unittest discover -s tests -v

