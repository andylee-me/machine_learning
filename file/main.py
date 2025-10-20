"""
優化模型決策邊界視覺化
展示準確率81%+的模型的決策邊界
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️  UMAP未安裝")

plt.rcParams['font.sans-serif'] = ['Microsoft YhHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("="*100)
print("優化模型決策邊界視覺化".center(100))
print("="*100)

# ============================================================================
# 1. 資料載入與特徵工程
# ============================================================================
print("\n【準備資料】")
df = pd.read_csv('spotify_churn_dataset.csv')
df_processed = df.drop(['user_id'], axis=1)

# 激進特徵工程
df_processed['engagement_score'] = (
    df_processed['listening_time'] * df_processed['songs_played_per_day'] / 
    (df_processed['skip_rate'] + 0.01)
)
df_processed['ad_burden'] = df_processed['ads_listened_per_week'] / (df_processed['listening_time'] + 1)
df_processed['usage_depth'] = df_processed['listening_time'] * df_processed['offline_listening']
df_processed['skip_to_listening_ratio'] = df_processed['skip_rate'] * df_processed['listening_time']
df_processed['songs_per_hour'] = df_processed['songs_played_per_day'] / (df_processed['listening_time'] / 60 + 1)
df_processed['ad_tolerance'] = 1 / (df_processed['ads_listened_per_week'] + 1)
df_processed['activity_level'] = df_processed['listening_time'] * df_processed['songs_played_per_day']
df_processed['skip_severity'] = df_processed['skip_rate'] ** 2
df_processed['engagement_to_skip'] = df_processed['engagement_score'] / (df_processed['skip_rate'] + 0.01)
df_processed['age_group'] = pd.cut(df_processed['age'], bins=[0, 25, 35, 45, 100], labels=[0, 1, 2, 3]).astype(int)
df_processed['listening_x_offline'] = df_processed['listening_time'] * df_processed['offline_listening']
df_processed['skip_x_ads'] = df_processed['skip_rate'] * df_processed['ads_listened_per_week']

for col in ['gender', 'country', 'subscription_type', 'device_type']:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col].astype(str))

df_processed['is_premium'] = (df_processed['subscription_type'] == 1).astype(int)
df_processed['premium_engagement'] = df_processed['is_premium'] * df_processed['engagement_score']

X = df_processed.drop('is_churned', axis=1).values
y = df_processed['is_churned'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用SMOTE-Tomek平衡
smote_tomek = SMOTETomek(random_state=42)
X_balanced, y_balanced = smote_tomek.fit_resample(X_scaled, y)

print(f"✓ 特徵數: {X_balanced.shape[1]}")
print(f"✓ 平衡後樣本: {X_balanced.shape[0]}")

# ============================================================================
# 2. 選擇最重要的3個特徵
# ============================================================================
print("\n【特徵選擇】")
feature_names = df_processed.drop('is_churned', axis=1).columns

selector = SelectKBest(mutual_info_classif, k='all')
selector.fit(X_balanced, y_balanced)
scores = selector.scores_

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'score': scores
}).sort_values('score', ascending=False)

print("✓ 前5重要特徵:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['feature']}: {row['score']:.3f}")

top3_features = feature_importance.head(3)['feature'].values
top3_indices = [list(feature_names).index(f) for f in top3_features]
X_top3 = X_balanced[:, top3_indices]

# ============================================================================
# 3. 降維
# ============================================================================
print("\n【降維】")

pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_balanced)
print(f"✓ PCA 2D: {sum(pca_2d.explained_variance_ratio_)*100:.1f}%")

pca_3d = PCA(n_components=3, random_state=42)
X_pca_3d = pca_3d.fit_transform(X_balanced)

tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne_2d = tsne_2d.fit_transform(X_balanced)
print(f"✓ t-SNE 2D完成")

tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30, max_iter=1000)
X_tsne_3d = tsne_3d.fit_transform(X_balanced)

if UMAP_AVAILABLE:
    umap_2d = UMAP(n_components=2, random_state=42)
    X_umap_2d = umap_2d.fit_transform(X_balanced)
    print(f"✓ UMAP完成")

# ============================================================================
# 4. 定義優化模型
# ============================================================================
classifiers = {
    '優化隨機森林': RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=10, min_samples_leaf=5,
        max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1
    ),
    '梯度提升樹': GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=7, min_samples_split=10,
        min_samples_leaf=5, subsample=0.8, random_state=42
    ),
    '深度決策樹': DecisionTreeClassifier(
        max_depth=15, min_samples_split=10, min_samples_leaf=5,
        class_weight='balanced', random_state=42
    )
}

# 集成模型
ensemble = VotingClassifier(
    estimators=[(name, clf) for name, clf in classifiers.items()],
    voting='soft'
)
classifiers['集成模型'] = ensemble

method_names = list(classifiers.keys())
colors = ['#3498DB', '#2ECC71', '#F39C12', '#E74C3C']

# ============================================================================
# 5. 繪圖函數
# ============================================================================
def plot_2d(ax, X_2d, y, clf, title, xlabel='X', ylabel='Y'):
    """2D決策邊界"""
    clf.fit(X_2d, y)
    
    h = 0.02
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.2, cmap='RdYlBu_r', levels=[0, 0.5, 1])
    ax.contour(xx, yy, Z, colors='black', linewidths=0.6, levels=[0.5], alpha=0.7)
    
    ax.scatter(X_2d[y==0, 0], X_2d[y==0, 1], c='#4169E1', marker='o', 
              s=6, alpha=0.3, edgecolors='none', label='未流失')
    ax.scatter(X_2d[y==1, 0], X_2d[y==1, 1], c='#DC143C', marker='s',
              s=6, alpha=0.3, edgecolors='none', label='已流失')
    
    ax.set_xlabel(xlabel, fontsize=9, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=9, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.85)
    ax.grid(alpha=0.12, linewidth=0.3)
    
    y_pred = clf.predict(X_2d)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    # 高亮顯示高準確率
    box_color = 'lightgreen' if acc >= 0.8 else 'yellow'
    ax.text(0.02, 0.98, f'Acc: {acc:.3f}\nF1: {f1:.3f}', transform=ax.transAxes, 
            fontsize=8, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.7, 
            edgecolor='darkgreen' if acc >= 0.8 else 'black', linewidth=1.5),
            verticalalignment='top')

def plot_3d(ax, X_3d, y, clf, title, xlabel='X', ylabel='Y', zlabel='Z'):
    """3D決策邊界"""
    clf.fit(X_3d, y)
    
    ax.scatter(X_3d[y==0, 0], X_3d[y==0, 1], X_3d[y==0, 2], 
              c='#4169E1', marker='o', s=3, alpha=0.2, 
              edgecolors='none', label='未流失', depthshade=True)
    ax.scatter(X_3d[y==1, 0], X_3d[y==1, 1], X_3d[y==1, 2], 
              c='#DC143C', marker='s', s=3, alpha=0.2, 
              edgecolors='none', label='已流失', depthshade=True)
    
    # 決策平面
    x_min, x_max = X_3d[:, 0].min(), X_3d[:, 0].max()
    y_min, y_max = X_3d[:, 1].min(), X_3d[:, 1].max()
    z_min, z_max = X_3d[:, 2].min(), X_3d[:, 2].max()
    
    for z_val in [z_min + (z_max - z_min) * i / 4 for i in range(1, 4)]:
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 25), np.linspace(y_min, y_max, 25))
        zz = np.full_like(xx, z_val)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()]).reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=[0.5], zdir='z', offset=z_val,
                  colors='purple', linewidths=0.6, alpha=0.3, linestyles='--')
    
    ax.set_xlabel(xlabel, fontsize=8, fontweight='bold', labelpad=5)
    ax.set_ylabel(ylabel, fontsize=8, fontweight='bold', labelpad=5)
    ax.set_zlabel(zlabel, fontsize=8, fontweight='bold', labelpad=5)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
    ax.legend(fontsize=6, loc='upper right', framealpha=0.8)
    ax.view_init(elev=20, azim=45)
    ax.grid(alpha=0.12, linewidth=0.3)
    ax.tick_params(labelsize=6)
    
    y_pred = clf.predict(X_3d)
    acc, f1 = accuracy_score(y, y_pred), f1_score(y, y_pred)
    box_color = 'lightgreen' if acc >= 0.8 else 'yellow'
    ax.text2D(0.02, 0.98, f'Acc: {acc:.3f}\nF1: {f1:.3f}', transform=ax.transAxes, 
              fontsize=7, fontweight='bold',
              bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.7,
              edgecolor='darkgreen' if acc >= 0.8 else 'black', linewidth=1.5),
              verticalalignment='top')

# ============================================================================
# 圖A: 2D決策邊界比較
# ============================================================================
print("\n【生成視覺化】")
print("✓ 圖A: 2D決策邊界...")

fig_a = plt.figure(figsize=(24, 18))
gs_a = fig_a.add_gridspec(4, 4, hspace=0.4, wspace=0.35)

# 第一行：PCA降維 - 4種模型
count = 0
for idx, (name, clf, color) in enumerate(zip(method_names, classifiers.values(), colors)):
    ax = fig_a.add_subplot(gs_a[0, idx])
    plot_2d(ax, X_pca_2d, y_balanced, clf, f'{name}\nPCA降維', 'PC1', 'PC2')
print("PCA降維 - 4種模型")

# 第二行：t-SNE降維 - 4種模型
for idx, (name, clf, color) in enumerate(zip(method_names, classifiers.values(), colors)):
    ax = fig_a.add_subplot(gs_a[1, idx])
    plot_2d(ax, X_tsne_2d, y_balanced, clf, f'{name}\nt-SNE降維', 't-SNE 1', 't-SNE 2')
    print("已經跑了",count,"次")
    count+=1
    
print("t-SNE降維 - 4種模型")

# 第三行：UMAP降維 - 4種模型
if UMAP_AVAILABLE:
    for idx, (name, clf, color) in enumerate(zip(method_names, classifiers.values(), colors)):
        ax = fig_a.add_subplot(gs_a[2, idx])
        plot_2d(ax, X_umap_2d, y_balanced, clf, f'{name}\nUMAP降維', 'UMAP 1', 'UMAP 2')
print("UMAP降維 - 4種模型")

# 第四行：前2特徵 - 4種模型
X_top2 = X_balanced[:, top3_indices[:2]]
f1_name = top3_features[0][:10] + '..' if len(top3_features[0]) > 10 else top3_features[0]
f2_name = top3_features[1][:10] + '..' if len(top3_features[1]) > 10 else top3_features[1]
print("前2特徵 - 4種模型")

for idx, (name, clf, color) in enumerate(zip(method_names, classifiers.values(), colors)):
    ax = fig_a.add_subplot(gs_a[3, idx])
    plot_2d(ax, X_top2, y_balanced, clf, f'{name}\n前2重要特徵', f1_name, f2_name)

fig_a.suptitle('優化模型決策邊界視覺化 (2D) - 準確率81%+', fontsize=22, fontweight='bold', y=0.995)
plt.savefig('優化模型決策邊界2D.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================================
# 圖B: 3D決策邊界
# ============================================================================
print("✓ 圖B: 3D決策邊界...")

fig_b = plt.figure(figsize=(24, 18))
gs_b = fig_b.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

# 第一行：PCA 3D - 4種模型
for idx, (name, clf) in enumerate(zip(method_names, classifiers.values())):
    ax = fig_b.add_subplot(gs_b[0, idx], projection='3d')
    plot_3d(ax, X_pca_3d, y_balanced, clf, f'{name}\nPCA 3D', 'PC1', 'PC2', 'PC3')

# 第二行：t-SNE 3D - 4種模型
for idx, (name, clf) in enumerate(zip(method_names, classifiers.values())):
    ax = fig_b.add_subplot(gs_b[1, idx], projection='3d')
    plot_3d(ax, X_tsne_3d, y_balanced, clf, f'{name}\nt-SNE 3D', 't-SNE 1', 't-SNE 2', 't-SNE 3')

# 第三行：前3特徵 3D - 4種模型
f1 = top3_features[0][:8] + '..' if len(top3_features[0]) > 8 else top3_features[0]
f2 = top3_features[1][:8] + '..' if len(top3_features[1]) > 8 else top3_features[1]
f3 = top3_features[2][:8] + '..' if len(top3_features[2]) > 8 else top3_features[2]

for idx, (name, clf) in enumerate(zip(method_names, classifiers.values())):
    ax = fig_b.add_subplot(gs_b[2, idx], projection='3d')
    plot_3d(ax, X_top3, y_balanced, clf, f'{name}\n前3特徵', f1, f2, f3)

fig_b.suptitle('優化模型決策邊界視覺化 (3D) - 準確率81%+', fontsize=22, fontweight='bold', y=0.995)
plt.savefig('優化模型決策邊界3D.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("\n" + "="*100)
print("完成!".center(100))
print("="*100)
print("\n✅ 已生成2張決策邊界圖")
print("  - 優化模型決策邊界2D.png (16張子圖)")
print("  - 優化模型決策邊界3D.png (12張子圖)")
print("\n💡 綠色框 = 準確率 ≥ 80%")
print("💡 黃色框 = 準確率 < 80%")
