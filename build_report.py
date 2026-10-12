# -*- coding: utf-8 -*-
# 由 outputs/results.json 与 figures/ 生成实验报告 PDF

import json
import os
import re

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
OUT = os.path.join(HERE, "outputs")
PDF_PATH = os.path.join(HERE, "实验一_文本分类_实验报告.pdf")

pdfmetrics.registerFont(TTFont("SimSun", r"C:\Windows\Fonts\simsun.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("SimHei", r"C:\Windows\Fonts\simhei.ttf"))

ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["Normal"], fontName="SimSun", fontSize=10.5,
                      leading=16, firstLineIndent=21, spaceAfter=4)
plain = ParagraphStyle("plain", parent=body, firstLineIndent=0)
h1 = ParagraphStyle("h1", parent=ss["Normal"], fontName="SimHei", fontSize=14,
                    leading=20, spaceBefore=10, spaceAfter=6)
h2 = ParagraphStyle("h2", parent=ss["Normal"], fontName="SimHei", fontSize=11.5,
                    leading=18, spaceBefore=6, spaceAfter=4)
title = ParagraphStyle("title", parent=ss["Title"], fontName="SimHei", fontSize=18,
                       leading=26, alignment=TA_CENTER)
sub = ParagraphStyle("sub", parent=ss["Normal"], fontName="SimSun", fontSize=11,
                     leading=18, alignment=TA_CENTER, textColor=colors.HexColor("#333333"))
cap = ParagraphStyle("cap", parent=ss["Normal"], fontName="SimSun", fontSize=9,
                     leading=13, alignment=TA_CENTER, textColor=colors.HexColor("#555555"))

with open(os.path.join(OUT, "results.json"), encoding="utf-8") as f:
    R = json.load(f)


def tcell(text, bold=False):
    st = ParagraphStyle("c", parent=ss["Normal"], fontName="SimHei" if bold else "SimSun",
                        fontSize=9, leading=12, alignment=1)
    return Paragraph(str(text), st)


def make_table(data, widths, header=True):
    t = Table(data, colWidths=widths, hAlign="CENTER", repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 3),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf5")))
    t.setStyle(TableStyle(style))
    return t


def parse_report(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", line)
        if m:
            rows.append(m.groups())
    return rows


S = []

S.append(Paragraph("当代人工智能 实验一 实验报告", title))
S.append(Spacer(1, 4))
S.append(Paragraph("基于 TF-IDF 与经典机器学习算法的文本分类", sub))
S.append(Spacer(1, 10))
S.append(make_table([
    [tcell("姓名"), tcell("银启行"), tcell("学号"), tcell("10245501485")],
    [tcell("课程"), tcell("当代人工智能"), tcell("日期"), tcell("2026 年 10 月")],
], [2.0 * cm, 5.5 * cm, 2.0 * cm, 5.5 * cm], header=False))
S.append(Spacer(1, 10))

S.append(Paragraph("一、实验目的", h1))
S.append(Paragraph(
    "本次实验完成一次完整的文本分类流程：数据加载与探索、文本预处理与 TF-IDF 特征表示、"
    "使用多种分类算法训练模型、在验证集上比较不同模型和不同参数的效果，"
    "最后挑选验证集表现最好的方案对无标签测试集进行预测。", body))
S.append(Paragraph(
    "实验中涉及随机的环节（训练/验证集划分、模型初始化）都固定了随机种子 random_state=42，"
    "保证结果可复现。", body))

S.append(Paragraph("二、实验数据与实验设置", h1))
S.append(Paragraph(
    f"数据是 20 Newsgroups 的 10 类新闻文本子集，共 {R['n_train'] + R['n_valid']} 条有标签样本、"
    f"{R['n_test']} 条无标签测试样本，类别编号 0~9。各类别样本数在 700 上下，分布基本均衡"
    "（图 1），所以不做额外的重采样。", body))
cc = R["class_counts"]
S.append(Paragraph("各类别样本数：" + "、".join(f"{k}：{v}" for k, v in cc.items()) + "。", plain))
S.append(Spacer(1, 4))
S.append(Image(os.path.join(FIG, "fig1_class_distribution.png"), width=11.5 * cm,
               height=6.1 * cm))
S.append(Paragraph("图 1  训练集各类别样本分布", cap))
S.append(Paragraph(
    "把有标签数据按 8:2 分层划分（train_test_split，stratify=y，random_state=42），"
    f"得到训练集 {R['n_train']} 条、验证集 {R['n_valid']} 条。验证集只用于模型比较和调参，"
    "不参与训练，避免信息泄露。特征使用 TF-IDF。", body))

S.append(Paragraph("三、模型方法", h1))
S.append(Paragraph("实验用到四种经典算法，都直接在 TF-IDF 特征上训练：", body))
for name, desc in [
    ("朴素贝叶斯（MultinomialNB）", "基于贝叶斯定理、假设各特征相互独立，计算简单、速度快，"
     "是文本分类常用的基线。参数 alpha 是平滑系数。"),
    ("线性支持向量机（SVC, kernel='linear'）", "寻找能把两类尽可能分开的最大间隔超平面；"
     "文本特征高维且稀疏，线性核通常效果好、速度快。参数 C 是错分的惩罚强度。"),
    ("逻辑回归（LogisticRegression）", "线性分类器，会给出各类别的概率。"
     "参数 C 是正则化强度的倒数，C 越小正则化越强。"),
    ("多层感知机（MLPClassifier）", "带隐层的神经网络，能学习非线性关系。"
     "参数有隐层结构 hidden_layer_sizes、L2 正则 alpha 和迭代次数 max_iter。"),
]:
    S.append(Paragraph(f"<b>{name}</b>：{desc}", body))

S.append(Paragraph("四、实验过程与结果", h1))
S.append(Paragraph("4.1 TF-IDF 特征表示对比", h2))
S.append(Paragraph(
    "为了单独观察特征表示的影响，固定用线性 SVM（C=1）作为“探针模型”，只在训练集上拟合、"
    "在验证集上评估。四个配置逐步加入去停用词、去重音、sublinear_tf（对词频取对数，抑制高频词）"
    "和二元短语（1,2）。结果见表 1。", body))
rows = [[tcell("配置", True), tcell("特征数", True), tcell("准确率", True),
         tcell("Macro-F1", True)]]
for v in R["vectorizer_results"]:
    rows.append([tcell(v["vectorizer"]), tcell(v["n_features"]),
                 tcell(f"{v['accuracy']:.4f}"), tcell(f"{v['macro_f1']:.4f}")])
S.append(make_table(rows, [7.6 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm]))
S.append(Paragraph("表 1  不同 TF-IDF 配置的验证集结果（线性 SVM 探针）", cap))
S.append(Spacer(1, 6))
S.append(Image(os.path.join(FIG, "fig2_vectorizer.png"), width=11.5 * cm, height=6.1 * cm))
S.append(Paragraph("图 2  不同 TF-IDF 配置的验证集表现", cap))
best_v = R["best_vectorizer"]
S.append(Paragraph(
    f"从表 1 看到，配置越好探针模型的准确率越高：V0 基线只有 0.9030，"
    f"V3（max_features=20000、(1,2) 二元短语 + sublinear_tf + 去停用词）达到 {best_v['accuracy']:.4f}。"
    "原因是二元短语能保留“not good”这类单个词无法表达的搭配信息，sublinear_tf 也能削弱高频词的统治。"
    "后续模型统一使用 V3 配置。", body))

S.append(Paragraph("4.2 模型与超参数对比", h2))
S.append(Paragraph(
    "在 V3 特征上对四种算法分别设置超参数网格，用验证集比较（表 2 是每个算法的最优结果）。"
    "SVM 和逻辑回归的 C 取 [0.1, 1, 10, 100]，朴素贝叶斯 alpha 取 [0.01, 0.1, 0.5, 1.0]，"
    "MLP 比较了 (100,)、(200,)、(100,50) 三种结构和两个 alpha。", body))
best_by_model = {}
for r in R["model_results"]:
    best_by_model.setdefault(r["model"], r)
order = ["朴素贝叶斯", "SVM(线性核)", "逻辑回归", "多层感知机"]
rows = [[tcell("算法", True), tcell("最优参数", True), tcell("准确率", True),
         tcell("Macro-F1", True), tcell("训练耗时/s", True)]]
for m in order:
    r = best_by_model[m]
    rows.append([tcell(m), tcell(r["params"]), tcell(f"{r['accuracy']:.4f}"),
                 tcell(f"{r['macro_f1']:.4f}"), tcell(r["train_time_s"])])
S.append(make_table(rows, [2.8 * cm, 5.4 * cm, 2.3 * cm, 2.4 * cm, 2.3 * cm]))
S.append(Paragraph("表 2  各算法最优配置的验证集结果", cap))
S.append(Spacer(1, 6))
S.append(Image(os.path.join(FIG, "fig3_models.png"), width=11.0 * cm, height=5.9 * cm))
S.append(Paragraph("图 3  各算法最优配置对比", cap))
S.append(Spacer(1, 4))
S.append(Image(os.path.join(FIG, "fig4_hyperparams.png"), width=15.5 * cm, height=4.5 * cm))
S.append(Paragraph("图 4  超参数对验证集 Macro-F1 的影响", cap))
S.append(Paragraph(
    "结果说明：（1）朴素贝叶斯最弱（Macro-F1 约 0.907），但训练几乎不耗时，是很好的快速基线；"
    "（2）线性 SVM 和逻辑回归很接近，都在 0.936 以上，SVM 在 C=10 时最好（0.9360），"
    "C 太小（0.1）时明显欠拟合（0.8514），C=100 时略有回落；"
    "（3）逻辑回归的 C 越大越好，C=100 达到 0.9377，训练只用约 1.5 秒；"
    "（4）MLP 在 (100,)、alpha=0.001 时取得最好成绩 0.9392，但训练要 179 秒。", body))
S.append(Paragraph(
    "把性能和复杂度放在一起看：MLP 只比逻辑回归高约 0.15 个百分点，却慢了两个数量级。"
    "如果更看重效率，逻辑回归 C=100 很有性价比；不过按实验“选最好的结果”的要求，"
    "最终提交仍采用验证集最优的 MLP (100,), alpha=0.001。", body))

doc = SimpleDocTemplate(PDF_PATH, pagesize=A4, topMargin=2 * cm, bottomMargin=1.8 * cm,
                        leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                        title="当代人工智能 实验一 实验报告")

S.append(Paragraph("4.3 最优模型分析", h2))
best = R["best_model"]
S.append(Paragraph(
    f"最优模型为<b>{best['model']}（{best['params']}）</b>，验证集准确率 {best['accuracy']:.4f}、"
    f"Macro-F1 {best['macro_f1']:.4f}。表 3 是它在验证集上的分类报告，图 5 是混淆矩阵。", body))
rows = [[tcell("类别", True), tcell("精确率", True), tcell("召回率", True),
         tcell("F1", True), tcell("样本数", True)]]
for g in parse_report(os.path.join(OUT, "classification_report.txt")):
    rows.append([tcell(g[0]), tcell(g[1]), tcell(g[2]), tcell(g[3]), tcell(g[4])])
S.append(make_table(rows, [2.2 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm]))
S.append(Paragraph("表 3  最优模型在验证集上的分类报告", cap))
S.append(Spacer(1, 6))
S.append(Image(os.path.join(FIG, "fig5_confusion_matrix.png"), width=10.5 * cm,
               height=8.9 * cm))
S.append(Paragraph("图 5  最优模型的混淆矩阵", cap))

cm = np.array(R["confusion_matrix"])
off = [(int(cm[i, j]), i, j) for i in range(len(cm)) for j in range(len(cm))
       if i != j and cm[i, j] > 0]
off.sort(reverse=True)
conf_txt = "、".join(f"真实{i}→预测{j}（{n} 次）" for n, i, j in off[:3])
S.append(Paragraph(
    f"大部分类别的 F1 在 0.90 以上，其中类别 9、5、4 最好（0.98 左右），"
    f"类别 2 和类别 7 相对偏低（F1 分别为 0.8350 和 0.8828）。"
    f"混淆主要集中在语义相近的类别上，最典型的错误是 {conf_txt}，"
    "基本都发生在类别 2 和类别 7 之间。这类样本主题相近、关键词重叠，"
    "只靠词频特征较难区分，是后续可以改进的方向（例如引入词序信息或预训练语言模型）。", body))

S.append(Paragraph("五、结论与思考", h1))
S.append(Paragraph(
    "结论：在固定随机种子、统一验证集的前提下，通过对比 TF-IDF 的四种配置和四种算法的参数网格，"
    "最终选出 MLP (100,), alpha=0.001（验证集 Macro-F1 0.9392）用于测试集预测，"
    "结果写入 predictions.csv 提交。", body))
S.append(Paragraph("<b>思考 1：</b>为什么 SVM 和逻辑回归在这类文本数据上表现很好？", h2))
S.append(Paragraph(
    "文本经过 TF-IDF 后是几万维的高维稀疏向量，且大多数类别近似线性可分。"
    "线性 SVM 和逻辑回归的决策边界都是线性的，参数数量与特征数同阶，在高维稀疏数据上不容易过拟合，"
    "训练和预测也很快，所以本次实验里它们仅次于 MLP，并明显好于朴素贝叶斯。", body))
S.append(Paragraph("<b>思考 2：</b>MLP 的结构越大、迭代越多，效果一定越好吗？", h2))
S.append(Paragraph(
    "不一定。本次实验里 (100,50) 的双层网络反而比单层 (100,) 差（0.9341 对 0.9392），"
    "训练过程中也出现过收敛较慢的情况。网络越大、迭代越多，越容易拟合稀疏文本特征中的噪声，"
    "训练时间也成倍增加，所以需要结合验证集结果和 alpha 正则来权衡，不能一味加大模型。", body))
S.append(Paragraph("<b>思考 3：</b>除了 TF-IDF 还有哪些文本表示方法？", h2))
S.append(Paragraph(
    "常见的有词袋 + 计数、n-gram 表示、词向量（Word2Vec、GloVe，把词映射到稠密向量后做平均或池化），"
    "以及预训练语言模型（BERT 等，能结合上下文得到句子的深度表示）。"
    "它们在语义表达上一般强于 TF-IDF，但计算成本更高，需要更多数据和算力。", body))

S.append(Paragraph("六、可复现说明", h1))
S.append(Paragraph(
    "运行环境：Python 3.10，numpy / pandas / scikit-learn。把 data/ 下的 train_data.csv、"
    "test_data_unlabeled.csv 与脚本放在同一目录即可复现。", body))
S.append(Paragraph(
    "执行 python experiment.py 会完成全部训练并在根目录生成 predictions.csv（表 1、表 2 与分类报告"
    "保存在 outputs/）；执行 python make_figures.py 生成图表，再执行 python build_report.py "
    "生成本报告。所有随机过程固定 random_state=42，重复运行结果一致。", body))

doc.build(S)
print("report saved:", PDF_PATH)
