# 当代人工智能 实验一：文本分类

20 Newsgroups 10 类新闻文本分类。用 TF-IDF 提取特征，比较朴素贝叶斯、线性 SVM、
逻辑回归、多层感知机四种算法及其不同超参数，按验证集 Macro-F1 选出最优方案，
对无标签测试集预测并输出 `predictions.csv`。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `experiment.py` | 主程序：数据加载、划分、特征对比、模型与超参数搜索、预测 |
| `make_figures.py` | 生成报告中的图表（读取 `outputs/results.json`） |
| `build_report.py` | 生成实验报告 PDF |
| `data/train_data.csv` | 有标签训练数据（7368 条） |
| `data/test_data_unlabeled.csv` | 无标签测试数据（2457 条） |
| `predictions.csv` | 最终提交的预测结果 |
| `outputs/results.json` | 全部实验指标汇总 |
| `outputs/classification_report.txt` | 最优模型的分类报告 |
| `figures/` | 报告插图 |

## 运行环境

```
pip install numpy pandas scikit-learn matplotlib reportlab
```

## 运行步骤

```
python experiment.py     # 训练与预测，生成 predictions.csv（约 15 分钟，MLP 较慢）
python make_figures.py   # 生成图表
python build_report.py   # 生成实验报告 PDF
```

所有随机过程固定 `random_state=42`，重复运行结果一致。
