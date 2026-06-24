# 3D Point Cloud Course Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Build a reproducible course-assignment package for a lightweight 3D point cloud detection report based on the provided paper.

**Architecture:** Use a NumPy-only synthetic voxel experiment to compare a baseline teacher, a lightweight student, and a distillation-trained lightweight student. Generate a Chinese report that states the reproducibility boundary and compares local results with the paper public metrics.

**Tech Stack:** Python, NumPy, ReportLab, python-docx-compatible Markdown/LaTeX text outputs.

---

### Task 1: Tests first

**Files:**
- Create: tests/test_core.py

- [ ] Write tests for deterministic synthetic data generation.
- [ ] Write tests for parameter/FLOP counting.
- [ ] Write tests for distillation losses returning finite non-negative values.
- [ ] Run tests and confirm they fail because production modules do not exist yet.

### Task 2: Core NumPy modules

**Files:**
- Create: src/data.py
- Create: src/models.py
- Create: src/losses.py
- Create: src/metrics.py
- Create: src/train.py
- Create: src/run_experiment.py

- [ ] Implement synthetic 3D voxel data generation.
- [ ] Implement baseline MLP teacher.
- [ ] Implement lightweight grouped student.
- [ ] Implement supervised and distillation training loops.
- [ ] Implement evaluation metrics and JSON/Markdown result export.

### Task 3: Report

**Files:**
- Create: report/course_report.md
- Create: report/course_report.tex
- Create: report/build_report_pdf.py
- Create: report/course_report.pdf

- [ ] Write a Chinese student-style report with abstract, current situation, method, experiment, discussion, conclusion, and references.
- [ ] Include assignment context, paper method summary, local reproducible experiment, and honest limitations.
- [ ] Generate a PDF with Chinese font support.

### Task 4: Verification

**Files:**
- Read: all created files.

- [ ] Run python -m unittest discover -s tests -v.
- [ ] Run python src/run_experiment.py.
- [ ] Run python report/build_report_pdf.py.
- [ ] Confirm expected artifacts exist under outputs/ and report/.

