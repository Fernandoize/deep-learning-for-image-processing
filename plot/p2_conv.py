import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns


# 生成模拟图像数据
def generate_image_data(n_samples=1000, img_size=(32, 32)):
    X = np.random.normal(0, 1, (n_samples, img_size[0], img_size[1]))
    return X


# 模拟卷积操作生成高维特征
def extract_conv_features(images, n_filters=16, kernel_size=3):
    n_samples, h, w = images.shape
    features = []

    # 定义随机卷积核
    kernels = np.random.normal(0, 1, (n_filters, kernel_size, kernel_size))

    for img in images:
        # 对每个图像应用多个卷积核
        conv_outputs = []
        for kernel in kernels:
            # 2D卷积，模式'same'保持输出尺寸
            conv_result = convolve2d(img, kernel, mode='same')
            # 池化（简单取平均）
            pooled = np.mean(conv_result)
            conv_outputs.append(pooled)
        features.append(conv_outputs)

    return np.array(features)  # (n_samples, n_filters)


# 生成目标变量（模拟复杂的非线性关系）
def generate_targets(features):
    # 假设目标是特征的二次组合加上噪声
    weights = np.random.normal(0, 1, features.shape[1])
    y = np.sum(features ** 2 * weights, axis=1) + np.random.normal(0, 0.1, features.shape[0])
    return y


# 计算特征统计特性
def compute_feature_stats(features):
    norms = np.linalg.norm(features, axis=1)
    variances = np.var(features, axis=1)
    return {
        'mean_norm': np.mean(norms),
        'mean_var': np.mean(variances)
    }


# 回归评估
def evaluate_regression(features, y):
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return r2_score(y_test, y_pred)


# 主实验函数
def run_experiment(n_samples=1000, img_size=(32, 32),
                   n_filters_list=[8, 16, 32], kernel_size=3):
    results = []

    # 生成图像数据
    images = generate_image_data(n_samples, img_size)

    for n_filters in n_filters_list:
        # 提取卷积特征
        features = extract_conv_features(images, n_filters, kernel_size)

        # 生成目标
        y = generate_targets(features)

        # 计算特征统计
        stats = compute_feature_stats(features)

        # 评估回归性能
        r2 = evaluate_regression(features, y)

        results.append({
            'n_filters': n_filters,
            'r2_score': r2,
            'mean_norm': stats['mean_norm'],
            'mean_var': stats['mean_var']
        })

    return results


# 可视化结果
def plot_results(results):
    n_filters = [r['n_filters'] for r in results]
    r2_scores = [r['r2_score'] for r in results]
    mean_norms = [r['mean_norm'] for r in results]
    mean_vars = [r['mean_var'] for r in results]

    sns.set(style="whitegrid")

    # R2分数随滤波器数量变化
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=n_filters, y=r2_scores, size=mean_norms, hue=mean_vars)
    plt.xlabel('Number of Filters (Feature Dimension)')
    plt.ylabel('R2 Score')
    plt.title('Regression Performance vs. Feature Dimension')
    plt.legend(title='Mean Variance')
    plt.show()

    # R2分数与特征范数的关系
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=mean_norms, y=r2_scores, size=n_filters, hue=mean_vars)
    plt.xlabel('Mean Feature Norm')
    plt.ylabel('R2 Score')
    plt.title('Regression Performance vs. Feature Norm')
    plt.legend(title='Mean Variance')
    plt.show()


# 运行实验
if __name__ == "__main__":
    np.random.seed(42)

    print("运行实验...")
    results = run_experiment()

    # 打印结果
    for r in results:
        print(f"Filters: {r['n_filters']}, R2: {r['r2_score']:.4f}, "
              f"Norm: {r['mean_norm']:.4f}, Var: {r['mean_var']:.4f}")

    print("绘制结果...")
    plot_results(results)