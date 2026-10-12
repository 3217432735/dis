# -*- coding: utf-8 -*-
# 生成实验报告所需的图表，读取 outputs/results.json

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

with open(os.path.join(HERE, "outputs", "results.json"), encoding="utf-8") as f:
    R = json.load(f)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


# 图1 类别分布
cc = R["class_counts"]
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.bar([str(k) for k in cc], list(cc.values()), color="#4c72b0")
ax.set_xlabel("类别编号")
ax.set_ylabel("样本数")
ax.set_title("训练集各类别样本分布")
save(fig, "fig1_class_distribution.png")

# 图2 TF-IDF 配置对比
vr = R["vectorizer_results"]
labels = [v["vectorizer"].split(" ")[0] for v in vr]
acc = [v["accuracy"] for v in vr]
f1 = [v["macro_f1"] for v in vr]
x = np.arange(len(vr))
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.bar(x - 0.2, acc, 0.4, label="准确率", color="#4c72b0")
ax.bar(x + 0.2, f1, 0.4, label="Macro-F1", color="#dd8452")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0.85, 0.96)
ax.set_ylabel("验证集得分")
ax.set_title("不同 TF-IDF 配置（线性SVM探针）")
ax.legend()
save(fig, "fig2_vectorizer.png")

# 图3 各模型最优结果
mr = R["model_results"]
best_by_model = {}
for r in mr:
    best_by_model.setdefault(r["model"], r)
names = list(best_by_model)
acc = [best_by_model[n]["accuracy"] for n in names]
f1 = [best_by_model[n]["macro_f1"] for n in names]
x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.bar(x - 0.2, acc, 0.4, label="准确率", color="#4c72b0")
ax.bar(x + 0.2, f1, 0.4, label="Macro-F1", color="#dd8452")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim(0.85, 0.96)
ax.set_ylabel("验证集得分")
ax.set_title("各算法最优配置的验证集表现")
ax.legend()
save(fig, "fig3_models.png")

# 图4 超参数敏感性
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

def curve(model_name, key, ax, xlabel):
    rows = [r for r in mr if r["model"] == model_name]
    xs, ys = [], []
    for r in rows:
        kv = dict(item.split("=") for item in r["params"].split(", "))
        xs.append(float(kv[key]))
        ys.append(r["macro_f1"])
    order = np.argsort(xs)
    xs = [xs[i] for i in order]
    ys = [ys[i] for i in order]
    ax.plot(xs, ys, "o-", color="#4c72b0")
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Macro-F1")
    ax.set_title(model_name)

curve("SVM(线性核)", "C", axes[0], "C（对数）")
curve("逻辑回归", "C", axes[1], "C（对数）")
curve("朴素贝叶斯", "alpha", axes[2], "alpha（对数）")
fig.suptitle("超参数对验证集 Macro-F1 的影响")
save(fig, "fig4_hyperparams.png")

# 图5 混淆矩阵
cm = np.array(R["confusion_matrix"])
fig, ax = plt.subplots(figsize=(5.2, 4.4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xlabel("预测类别")
ax.set_ylabel("真实类别")
ax.set_title(f"最优模型混淆矩阵（{R['best_model']['model']}）")
ax.set_xticks(range(len(cm)))
ax.set_yticks(range(len(cm)))
for i in range(len(cm)):
    for j in range(len(cm)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=7, color="black" if cm[i, j] < cm.max() * 0.6 else "white")
fig.colorbar(im, ax=ax, fraction=0.046)
save(fig, "fig5_confusion_matrix.png")

print("figures done")
