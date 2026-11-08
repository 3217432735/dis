# 当代人工智能 实验二：搜索算法

本项目实现迷宫 A*、加权地形 A*、极大极小树搜索与 Alpha-Beta
剪枝，并完成宝藏分配综合题。

## 文件

| 文件 | 内容 |
| --- | --- |
| `search_algorithms.py` | 算法与综合题实现 |
| `experiment.py` | 固定用例、对照实验与结果输出 |
| `tests/test_algorithms.py` | 单元测试 |
| `build_report.py` | 生成实验报告 PDF |
| `outputs/results.json` | 实验数据 |

## 运行

```bash
pip install -r requirements.txt
python experiment.py
python -m unittest discover -s tests -v
python build_report.py
```

固定用例的输出应分别为：迷宫路径长度 7、加权地形代价 7、
博弈树得分 5，以及 H=2 选择 T2、H=4 选择 T3。
