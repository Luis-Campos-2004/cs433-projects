"""As seis funções obrigatórias do Projeto 1 do CS-433.

Assinaturas e valores de retorno seguem o formato pedido habitualmente:
cada função devolve `(w, loss)`, em que `loss` é a perda calculada com o `w`
FINAL. Confirma os nomes exatos na descrição oficial deste ano antes de
submeter — a grelha de correção é automática e não perdoa typos.

Restrição típica do Projeto 1: só numpy. Nada de scikit-learn.
"""

import numpy as np

# ---------------------------------------------------------------- perdas


def compute_mse(y, tx, w):
    """Perda de erro quadrático médio: 1/(2N) * sum(e^2)."""
    e = y - tx @ w
    return 0.5 * np.mean(e**2)


def compute_gradient_mse(y, tx, w):
    """Gradiente do MSE."""
    e = y - tx @ w
    return -tx.T @ e / len(y)


def sigmoid(t):
    """Sigmoide numericamente estável (sem overflow para |t| grande)."""
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t)
    pos = t >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-t[pos]))
    exp_t = np.exp(t[~pos])
    out[~pos] = exp_t / (1.0 + exp_t)
    return out


def compute_logistic_loss(y, tx, w):
    """Log-verosimilhança negativa média. Espera y em {0, 1}."""
    z = tx @ w
    return np.mean(np.logaddexp(0, z) - y * z)


def compute_logistic_gradient(y, tx, w):
    """Gradiente da perda logística."""
    return tx.T @ (sigmoid(tx @ w) - y) / len(y)


# ------------------------------------------------- funções obrigatórias


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Regressão linear por gradiente descendente."""
    w = initial_w
    for _ in range(max_iters):
        w = w - gamma * compute_gradient_mse(y, tx, w)
    return w, compute_mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Regressão linear por SGD com batch de tamanho 1."""
    w = initial_w
    for _ in range(max_iters):
        i = np.random.randint(len(y))
        grad = compute_gradient_mse(y[i : i + 1], tx[i : i + 1], w)
        w = w - gamma * grad
    return w, compute_mse(y, tx, w)


def least_squares(y, tx):
    """Solução exata das equações normais."""
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Regressão ridge por equações normais."""
    n, d = tx.shape
    a = tx.T @ tx + 2 * n * lambda_ * np.eye(d)
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    return w, compute_mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Regressão logística por gradiente descendente. y em {0, 1}."""
    w = initial_w
    for _ in range(max_iters):
        w = w - gamma * compute_logistic_gradient(y, tx, w)
    return w, compute_logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regressão logística regularizada (L2). A perda devolvida NÃO inclui
    o termo de penalização — atenção a este detalhe, é onde toda a gente cai."""
    w = initial_w
    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
    return w, compute_logistic_loss(y, tx, w)
