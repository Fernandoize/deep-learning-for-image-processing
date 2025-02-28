import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns


# 生成不同分布的高维数据
def generate_data(n_samples=1000, n_features=50, distribution='normal'):
    if distribution == 'normal':
        X = np.random.normal(0, 1, (n_samples, n_features))
    elif distribution == 'uniform':
        X = np.random.uniform(-1, 1, (n_samples, n_features))
    elif distribution == 'exponential':
        X = np.random.exponential(1, (n_samples, n_features))
    else:
        raise ValueError("Unsupported distribution")

    # 生成目标变量（模拟一个线性关系加上噪声）
    true_weights = np.random.normal(0, 1, n_features)
    y = X.dot(true_weights) + np.random.normal(0, 0.1, n_samples)

    return X, y


# 计算特征向量的统计特性
def compute_vector_stats(X):
    # 计算每一维的均值和方差
    mean_per_dim = np.mean(X, axis=0)
    var_per_dim = np.var(X, axis=0)

    # 计算向量的模（L2范数）
    norms = np.linalg.norm(X, axis=1)

    return {
        'mean_mean': np.mean(mean_per_dim),
        'mean_var': np.mean(var_per_dim),
        'mean_norm': np.mean(norms)
    }


# 训练回归模型并评估
def evaluate_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return r2_score(y_test, y_pred)


# 主实验函数
def run_experiment(n_samples=1000, n_features_list=[10, 50, 100],
                   distributions=['normal', 'uniform', 'exponential']):
    results = []

    for n_features in n_features_list:
        for dist in distributions:
            # 生成数据
            X, y = generate_data(n_samples, n_features, dist)

            # 计算统计特性
            stats = compute_vector_stats(X)

            # 评估回归性能
            r2 = evaluate_regression(X, y)

            results.append({
                'n_features': n_features,
                'distribution': dist,
                'r2_score': r2,
                'mean_norm': stats['mean_norm'],
                'mean_var': stats['mean_var']
            })

    return results


# 可视化结果
def plot_results(results):
    # 转换为数组便于绘图
    n_features = [r['n_features'] for r in results]
    distributions = [r['distribution'] for r in results]
    r2_scores = [r['r2_score'] for r in results]
    mean_norms = [r['mean_norm'] for r in results]

    # 设置Seaborn风格
    sns.set(style="whitegrid")

    # R2分数随维度和分布变化
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=n_features, y=r2_scores, hue=distributions, size=mean_norms)
    plt.xlabel('Number of Features')
    plt.ylabel('R2 Score')
    plt.title('Regression Performance vs. Feature Dimension and Distribution')
    plt.legend(title='Distribution')
    plt.show()

    # R2分数与向量模的关系
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=mean_norms, y=r2_scores, hue=distributions, size=n_features)
    plt.xlabel('Mean Vector Norm')
    plt.ylabel('R2 Score')
    plt.title('Regression Performance vs. Vector Norm')
    plt.legend(title='Distribution')
    plt.show()


# 运行实验
if __name__ == "__main__":
    # 设置随机种子以保证可重复性
    np.random.seed(42)

    # 运行实验
    print("运行实验...")
    results = run_experiment()

    # 打印结果
    for r in results:
        print(f"Features: {r['n_features']}, Dist: {r['distribution']}, "
              f"R2: {r['r2_score']:.4f}, Norm: {r['mean_norm']:.4f}, "
              f"Var: {r['mean_var']:.4f}")

    # 可视化
    print("绘制结果...")
    plot_results(results)