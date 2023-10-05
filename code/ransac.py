import random
import numpy as np
import matplotlib.pyplot as plt

# 生成随机样本数据
x = np.arange(-5, 5, 0.1)
y = 5 * x ** 3 - 2 * x ** 2 + 3 * x + np.random.randn(len(x))

def fit_polynomial(x, y, degree):
    # 返回拟合多项式的系数
    return np.polyfit(x, y, degree)

def evaluate_polynomial(coef, x):
    # 计算多项式函数值
    return np.polyval(coef, x)

def ransac_polynomial(x, y, degree, n_iter, threshold):
    best_inliers = None
    best_coef = None
    best_err = np.inf
    for i in range(n_iter):
        # 随机选择若干个样本点
        sample_indices = random.sample(range(len(x)), degree + 1)
        sample_x = x[sample_indices]
        sample_y = y[sample_indices]

        # 拟合多项式
        coef = fit_polynomial(sample_x, sample_y, degree)

        # 计算所有样本点到多项式的距离
        all_errors = np.abs(evaluate_polynomial(coef, x) - y)

        # 选择符合阈值内的样本点
        inliers = all_errors < threshold
        num_inliers = np.sum(inliers)

        # 如果当前符合阈值的样本点数量比之前的多，则更新最佳参数
        if num_inliers > degree and num_inliers > np.sum(best_inliers):
            best_inliers = inliers
            best_coef = fit_polynomial(x[inliers], y[inliers], degree)
            best_err = np.sum(np.abs(evaluate_polynomial(best_coef, x[inliers]) - y[inliers]))
    
    return best_coef, best_err, best_inliers

if __name__ == "__main__":
    # 进行RANSAC拟合
    degree = 3
    n_iter = 100
    threshold = 0.5
    best_coef, best_err, best_inliers = ransac_polynomial(x, y, degree, n_iter, threshold)

    # 画出拟合曲线和数据点
    plt.plot(x, y, 'o')
    plt.plot(x[best_inliers], y[best_inliers], 'ro', alpha=0.5)
    plt.plot(x, evaluate_polynomial(best_coef, x), '-r', label='RANSAC', alpha=0.5)
    plt.plot(x, evaluate_polynomial(fit_polynomial(x, y, degree), x), '-g', label='Ordinary Least Squares')
    plt.legend()
    plt.show()
