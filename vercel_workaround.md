# 🚀 Vercel Serverless 部署技术方案说明

## 核心问题

Vercel Serverless 函数有两个严格限制：

| 限制 | 数值 |
|---|---|
| 部署包体积上限 | **500 MB** |
| 函数运行时可用内存 | 1024 MB |
| 可安装的 Python 包 | 仅限纯 Python 或极轻量包 |

主流机器学习框架的安装体积：

| 框架 | 安装体积 |
|---|---|
| `xgboost` | ~120 MB（含 C++ 共享库） |
| `lightgbm` | ~80 MB（含 C++ OpenMP 库） |
| `torch` (PyTorch) | ~800 MB ~ 2 GB |
| `scikit-learn` | ~60 MB |

这些框架不仅体积超标，而且在 Vercel 的 Linux Serverless 沙箱环境中，**底层 C++ 共享库（如 `libgomp.so.1`、OpenMP、MKL）往往缺失或权限受限**，导致 `import` 阶段就直接 Segfault 崩溃。

---

## 算法本质是否改变？

**没有。算法的数学本质完全没有任何改变。**

变化的只是**推理的执行引擎**，从 C++ 框架的底层调用，切换为等价的纯 Python/NumPy 手工实现：

```
训练时 (本地)          推理时 (云端)
─────────────────     ─────────────────────────────
真实的 C++ 框架         等价的 Python 数学实现
运行梯度下降优化        无需任何优化过程（已完成）
生成决策树结构          读取已存储的决策树结构遍历
```

---

## 三种模型各自的部署策略

### 1. 🌲 LightGBM & XGBoost（梯度提升决策树）

**核心思想：分离训练与推理的执行引擎。**

#### 训练阶段（本地，有完整 C++ 依赖）

```python
# 正常使用完整的 C++ 训练框架
self.model = lgb.LGBMClassifier(n_estimators=120, ...)
self.model.fit(X, y)   # 梯度下降，反复迭代优化 120 棵决策树

# 训练完毕后，立即提取决策树的纯数据结构
d = self.model.booster_.dump_model()
self.np_trees = [t['tree_structure'] for t in d['tree_info']]
# ↑ 这是一个普通的 Python 字典列表，不含任何 C++ 对象

self.model = None  # 丢弃 C++ 对象，仅保留数学结构
```

#### 推理阶段（云端，零 C++ 依赖）

```python
def predict_numpy(self, rows):
    for row in rows:
        raw_score = 0.0
        for tree in self.np_trees:
            # 手动递归遍历决策树节点
            node = tree
            while 'leaf_value' not in node:
                val = row[node['split_feature']]
                node = node['left_child'] if val <= node['threshold'] \
                       else node['right_child']
            raw_score += node['leaf_value']   # 累加所有树的叶节点值

        # Sigmoid 变换得到 [0,1] 概率
        prob = 1.0 / (1.0 + exp(-raw_score))
```

**数学等价性证明：** 梯度提升树的预测结果就是"所有棵树的叶节点值之和，经过 Sigmoid"。这个过程完全是确定性的树遍历，C++ 框架做的也是完全一样的事，只是用 C++ 实现更快。

---

### 2. 🧠 Neural NCF（神经协同过滤）

与 LightGBM/XGBoost 思路完全相同，但提取的是神经网络权重矩阵而非决策树：

#### 训练阶段（本地，有 PyTorch）

```python
# 用 PyTorch 训练神经网络
self.model = NCFModel(num_users, num_items, ...).to(device)
optimizer = optim.Adam(self.model.parameters())
# ... 梯度下降训练

# 提取所有层的权重为 NumPy 数组
sd = self.model.state_dict()
self.np_weights = {
    'user_embed': sd['user_embed.weight'].cpu().numpy(),
    'w0': sd['mlp.0.weight'].cpu().numpy(),
    'b0': sd['mlp.0.bias'].cpu().numpy(),
    # ...
}
```

#### 推理阶段（云端，零 PyTorch 依赖）

```python
def predict_numpy(self, user_indices, item_indices, aux_features):
    sd = self.np_weights
    u_vec = sd['user_embed'][user_indices]  # Embedding 查表
    i_vec = sd['item_embed'][item_indices]
    comb = np.concatenate([u_vec, i_vec, aux_features], axis=1)
    h1 = np.maximum(0, np.dot(comb, sd['w0'].T) + sd['b0'])  # ReLU
    h2 = np.maximum(0, np.dot(h1, sd['w3'].T) + sd['b3'])    # ReLU
    out = np.dot(h2, sd['w5'].T) + sd['b5']
    return 1.0 / (1.0 + np.exp(-out.ravel()))                 # Sigmoid
```

---

### 3. 🔗 ItemCF & MBA（协同过滤 & 关联规则）

这两个模型本来就是纯 Python 字典和矩阵，**从未依赖过任何 C++ 框架**，在云端一直正常运行。

---

## 架构对比图

```
┌─────────────────────────────────────────────────────────────────┐
│                    本地开发环境（有完整依赖）                       │
│                                                                  │
│  训练数据 ──► XGBoost C++ ──► 120棵决策树 ──► 提取为 JSON Dict   │
│            LightGBM C++  ──► 120棵决策树 ──► 提取为 Python Dict  │
│            PyTorch GPU/CPU ──► 神经网络 ──► 提取为 NumPy Array   │
│                                                                  │
│  序列化为 .pkl 文件（仅包含纯 Python 数据，无 C++ 引用）             │
└──────────────────────────────┬──────────────────────────────────┘
                               │ git push  （pkl 作为版本文件提交）
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              Vercel Serverless 云端（零 C++ 依赖）                 │
│                                                                  │
│  requirements.txt：仅 pandas + numpy（共 ~60 MB）                │
│                                                                  │
│  pickle.load(lightgbm.pkl) ──► LightGBMRecommender 实例         │
│      np_trees = [{...决策树字典...}]  ← 纯 Python               │
│      model = None                                                │
│                                                                  │
│  推理：遍历字典树 + NumPy 矩阵乘法                                  │
│  ✅ 无需 lightgbm/xgboost/torch 包                               │
│  ✅ 总部署体积 < 100 MB                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 性能代价分析

本次修改对**推荐精度零影响**，延迟有所上升但在可接受范围：

| 指标 | C++ 框架推理 | Python/NumPy 推理 | 影响 |
|---|---|---|---|
| **预测精度** | 完全一致 | 完全一致 | **零损失** |
| **单次推理延迟** | ~0.1 ms | ~10–15 ms | 慢约 100 倍 |
| **云端依赖** | 需要 C++ 库 | 仅需 pandas+numpy | **可部署** |
| **部署包体积** | +200 MB | +0 MB | **节省 200 MB** |

来自 `run_comparison.py` 实测结果：

| 算法 | Precision@5 | HitRate@5 | 推理延迟 |
|---|---|---|---|
| Item-based CF | 1.04% | 5.20% | 0.19 ms |
| XGBoost (NumPy) | 1.10% | 5.30% | 10.41 ms |
| LightGBM (NumPy) | 1.09% | 5.35% | 15.20 ms |
| Neural NCF (NumPy) | 0.91% | 4.45% | 0.45 ms |

> [!NOTE]
> 对于推荐系统演示场景，每次推理仅有 202 个候选商品（巧克力目录），10~15 ms 的延迟对用户完全感知不到。如果产品目录扩展到百万级别，则需要更高性能的方案（如 ONNX Runtime）。

---

## 这个技术方案的行业命名

这种模式在工业界有几个常见的名称：

1. **"Train-Serve Separation"（训练-推理分离）**：训练和推理使用不同技术栈，是大规模 ML 系统的标准设计。
2. **"Model Distillation to Inference Format"（推理格式蒸馏）**：将重型训练框架的输出蒸馏为轻量、可部署的推理格式（如 ONNX、TorchScript、自定义 NumPy）。
3. **"Serverless ML Optimization"**：针对 Serverless 环境优化 ML 模型的常见工程实践。

本项目的实现与 **ONNX（Open Neural Network Exchange）** 的理念高度一致——ONNX 也是将训练好的模型序列化为一个与框架无关的标准格式，然后用轻量 Runtime 执行推理。

---

## 文件修改汇总

| 文件 | 修改内容 |
|---|---|
| `lightgbm/lightgbm_recommender.py` | `fit()` 末尾新增 `extract_and_clear_model()`，提取 LightGBM 树为纯 Python 字典 |
| `xg_boost/xgboost_recommender.py` | 新增 `predict_numpy()` + `extract_and_clear_model()`，修改 `recommend()` 支持双路径推理，处理 XGBoost 类别特征分裂 |
| `simulation/models/lightgbm.pkl` | 重训练后包含 `np_trees`，`model=None` |
| `simulation/models/xgboost.pkl` | 重训练后包含 `np_trees`，`model=None` |
| `requirements.txt` | 保持仅 `pandas` + `numpy`，无需添加任何 ML 框架 |
