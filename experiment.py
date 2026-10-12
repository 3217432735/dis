# -*- coding: utf-8 -*-
# 当代人工智能 实验一：文本分类（20 Newsgroups，10 类）
# 用法：python experiment.py

import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=ConvergenceWarning)

SEED = 42
VALID_SIZE = 0.2
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "outputs")
CACHE_PATH = os.path.join(OUT_DIR, "fit_cache.json")
CM_PATH = os.path.join(OUT_DIR, "confusion_matrix.npy")
PRED_PATH = os.path.join(HERE, "predictions.csv")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_data.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_data_unlabeled.csv"))
    X = train_df["text"].astype(str).tolist()
    y = train_df["target"].values
    X_test = test_df["text"].astype(str).tolist()
    assert train_df["target"].isnull().sum() == 0
    assert train_df["text"].isnull().sum() == 0
    return X, y, X_test


def build_vectorizer(cfg):
    return TfidfVectorizer(
        max_features=cfg.get("max_features", None),
        ngram_range=cfg.get("ngram_range", (1, 1)),
        min_df=cfg.get("min_df", 1),
        sublinear_tf=cfg.get("sublinear_tf", False),
        stop_words=cfg.get("stop_words", None),
        strip_accents=cfg.get("strip_accents", None),
        lowercase=True,
    )


def evaluate(model, X_tr, y_tr, X_va, y_va, name, param_str, cache):
    key = f"{name}|{param_str}"
    if key in cache:
        rec = cache[key]
        log(f"  {name:<14} {param_str:<30} acc={rec['accuracy']:.4f}  "
            f"macroF1={rec['macro_f1']:.4f}  (缓存)")
        return rec
    t0 = time.time()
    model.fit(X_tr, y_tr)
    cost = time.time() - t0
    pred = model.predict(X_va)
    acc = accuracy_score(y_va, pred)
    f1m = f1_score(y_va, pred, average="macro")
    log(f"  {name:<14} {param_str:<30} acc={acc:.4f}  macroF1={f1m:.4f}  ({cost:.1f}s)")
    rec = {"model": name, "params": param_str, "accuracy": round(float(acc), 4),
           "macro_f1": round(float(f1m), 4), "train_time_s": round(float(cost), 2)}
    cache[key] = rec
    save_cache(cache)
    return rec


def make_model(best):
    name, p = best["model"], best["params"]
    kv = dict(item.split("=") for item in p.split(", "))
    if name == "朴素贝叶斯":
        return MultinomialNB(alpha=float(kv["alpha"]))
    if name == "SVM(线性核)":
        return SVC(kernel="linear", C=float(kv["C"]), random_state=SEED)
    if name == "逻辑回归":
        return LogisticRegression(C=float(kv["C"]), max_iter=2000, random_state=SEED)
    if name == "多层感知机":
        hidden = tuple(int(x) for x in kv["hidden"].strip("()").rstrip(",").split(","))
        return MLPClassifier(hidden_layer_sizes=hidden, alpha=float(kv["alpha"]),
                             max_iter=300, random_state=SEED)
    raise ValueError(name)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    log("步骤1 数据加载")
    X, y, X_test = load_data()
    classes = sorted(np.unique(y).tolist())
    class_counts = {int(c): int((y == c).sum()) for c in classes}
    log(f"训练样本 {len(X)}，测试样本 {len(X_test)}，类别数 {len(classes)}")
    log(f"各类别样本数: {class_counts}")

    log("步骤2 划分训练/验证集")
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=VALID_SIZE, random_state=SEED, stratify=y)
    log(f"训练集 {len(X_tr)}，验证集 {len(X_va)}")

    log("步骤3 TF-IDF 配置对比（用线性SVM做探针）")
    vec_configs = {
        "V0 基线 max_features=5000": dict(max_features=5000),
        "V1 去停用词+去重音": dict(max_features=5000, stop_words="english",
                                   strip_accents="unicode", min_df=2),
        "V2 V1+sublinear_tf": dict(max_features=5000, stop_words="english",
                                   strip_accents="unicode", min_df=2,
                                   sublinear_tf=True),
        "V3 V2+二元短语(1,2)": dict(max_features=20000, stop_words="english",
                                    strip_accents="unicode", min_df=2,
                                    sublinear_tf=True, ngram_range=(1, 2)),
    }
    vec_results = []
    for vname, cfg in vec_configs.items():
        vec = build_vectorizer(cfg)
        Xtr_v = vec.fit_transform(X_tr)
        Xva_v = vec.transform(X_va)
        probe = SVC(kernel="linear", C=1.0, random_state=SEED)
        probe.fit(Xtr_v, y_tr)
        pred = probe.predict(Xva_v)
        rec = {"vectorizer": vname, "n_features": int(Xtr_v.shape[1]),
               "accuracy": round(float(accuracy_score(y_va, pred)), 4),
               "macro_f1": round(float(f1_score(y_va, pred, average="macro")), 4),
               "config": cfg}
        vec_results.append(rec)
        log(f"  {vname:<26} 特征数={rec['n_features']:<7} "
            f"acc={rec['accuracy']:.4f}  macroF1={rec['macro_f1']:.4f}")
    best_vec = max(vec_results, key=lambda r: (r["macro_f1"], r["accuracy"]))
    log(f"最优向量化: {best_vec['vectorizer']}")

    log("步骤4 四种算法 × 超参数网格")
    vec = build_vectorizer(best_vec["config"])
    Xtr = vec.fit_transform(X_tr)
    Xva = vec.transform(X_va)
    results = []
    cache = load_cache()

    for alpha in [0.01, 0.1, 0.5, 1.0]:
        results.append(evaluate(MultinomialNB(alpha=alpha), Xtr, y_tr, Xva, y_va,
                                "朴素贝叶斯", f"alpha={alpha}", cache))
    for C in [0.1, 1, 10, 100]:
        results.append(evaluate(SVC(kernel="linear", C=C, random_state=SEED),
                                Xtr, y_tr, Xva, y_va, "SVM(线性核)", f"C={C}", cache))
    for C in [0.1, 1, 10, 100]:
        results.append(evaluate(LogisticRegression(C=C, max_iter=2000, random_state=SEED),
                                Xtr, y_tr, Xva, y_va, "逻辑回归", f"C={C}", cache))
    for hidden, alpha in [((100,), 0.0001), ((100,), 0.001),
                          ((200,), 0.001), ((100, 50), 0.001)]:
        results.append(evaluate(
            MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, max_iter=300,
                          random_state=SEED),
            Xtr, y_tr, Xva, y_va, "多层感知机", f"hidden={hidden}, alpha={alpha}", cache))

    results_sorted = sorted(results, key=lambda r: (r["macro_f1"], r["accuracy"]),
                            reverse=True)
    best = results_sorted[0]
    log("验证集排名前5:")
    for r in results_sorted[:5]:
        log(f"  {r['model']:<12} {r['params']:<30} "
            f"acc={r['accuracy']:.4f}  macroF1={r['macro_f1']:.4f}")

    def write_summary(cm_obj=None, report=None):
        s = {"seed": SEED, "valid_size": VALID_SIZE, "n_train": len(X_tr),
             "n_valid": len(X_va), "n_test": len(X_test), "n_classes": len(classes),
             "class_counts": class_counts, "vectorizer_results": vec_results,
             "best_vectorizer": best_vec, "model_results": results_sorted,
             "best_model": best}
        if cm_obj is not None:
            s["confusion_matrix"] = cm_obj.tolist()
        if report is not None:
            s["classification_report"] = report
        with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)

    write_summary()  # 先保存网格结果，避免中断丢失

    log("步骤5 最优模型的分类报告")
    if os.path.exists(CM_PATH) and os.path.exists(
            os.path.join(OUT_DIR, "classification_report.txt")):
        cm = np.load(CM_PATH)
        with open(os.path.join(OUT_DIR, "classification_report.txt"),
                  encoding="utf-8") as f:
            report_txt = f.read()
        log("  已存在，跳过重复训练")
    else:
        best_model = make_model(best)
        best_model.fit(Xtr, y_tr)
        y_pred_va = best_model.predict(Xva)
        cm = confusion_matrix(y_va, y_pred_va, labels=classes)
        report_txt = classification_report(y_va, y_pred_va, labels=classes,
                                           digits=4, zero_division=0)
        with open(os.path.join(OUT_DIR, "classification_report.txt"), "w",
                  encoding="utf-8") as f:
            f.write(f"最优模型: {best['model']}  ({best['params']})\n\n{report_txt}")
        np.save(CM_PATH, cm)
    write_summary(cm, report_txt)

    log("步骤6 全量重训并预测测试集")
    if os.path.exists(PRED_PATH):
        log("  已存在，跳过重复训练")
    else:
        vec_full = build_vectorizer(best_vec["config"])
        Xall = vec_full.fit_transform(X)
        Xtest = vec_full.transform(X_test)
        final_model = make_model(best)
        final_model.fit(Xall, y)
        test_pred = final_model.predict(Xtest).astype(int)
        pd.DataFrame(test_pred).to_csv(PRED_PATH, index=False, header=False)
        log(f"已保存 {PRED_PATH}（{len(test_pred)} 条）")

    write_summary(cm, report_txt)
    log("结果保存到 outputs/results.json，完成")


if __name__ == "__main__":
    main()
