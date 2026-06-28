# 🚀 XGBoost Ranking Recommender System (XGBoost 树精排模型)

本模块实现了基于 **XGBoost (Extreme Gradient Boosting)** 的工业级推荐系统精排模型，用于在召回候选集的基础上，进行高精确度的转化率 (CTR/CVR) 打分与重排序。

---

## 🎯 核心设计与工程特色

1. **精准结构化特征交叉**：
   - 整合用户年龄、性别、会员等级等人口统计学画像。
   - 提取商品可可百分比 (`cocoa_percent`) 与重量规格 (`weight_g`) 连续属性。
   - 拼接用户历史消费偏好统计量与销售渠道特征。

2. **高效梯度提升决策树**：
   - 采用二分类 Logloss (`binary:logistic`) 拟合购买点击行为。
   - 利用 `Hist` 树构建算法加速连续特征分箱计算。

3. **健壮的 OOV 兜底机制**：
   - 内部建立分类特征编码 Mapping 字典。
   - 针对缺失值与未见枚举属性，自动平滑映射至默认先验类别。

---

## 🛠️ 快速使用示例 (Python API)

```python
from xg_boost.xgboost_recommender import XGBoostRecommender

# 初始化并加载训练好的模型
recommender = XGBoostRecommender()
recommender.load_model("models/xgboost_model.json")

# 为指定用户在特定门店生成精排 Top 5 推荐列表
recommendations = recommender.recommend(user_id="CUST_000123", k=5, store_id="S001")
print("Top Recommended Product IDs:", recommendations)
```

---

## 📊 模型优势与商业定位

- **高度可解释性**：支持输出特征重要性排序 (Feature Importance Gain/Weight)。
- **精排备选与对比基线**：在工业级推荐流水线中，与 LightGBM 形成经典树模型对比。
