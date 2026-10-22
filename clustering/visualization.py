import matplotlib
matplotlib.use("Agg")  # lưu ra file PNG, không cần mở cửa sổ
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def plot_k_selection(scores, chosen_k, path, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(scores["k"], scores["inertia"], marker="o")
    axes[0].set_title("Elbow (inertia)")
    axes[1].plot(scores["k"], scores["silhouette"], marker="o", color="tab:green")
    axes[1].set_title("Silhouette (cosine) - càng cao càng tốt")
    axes[2].plot(scores["k"], scores["stability_ari"], marker="o", color="tab:red")
    axes[2].axhline(0.6, ls="--", color="gray")
    axes[2].set_title("Độ ổn định (ARI) - ngưỡng 0.6")
    for ax in axes:
        ax.axvline(chosen_k, ls=":", color="black")
        ax.set_xlabel("k")
    fig.suptitle(f"{title} - chọn k={chosen_k}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pca(X, labels, path, title):
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=labels.values, palette="tab10", s=70, ax=ax)
    ax.set_title(f"{title}\n(PCA 2 chiều, giải thích {pca.explained_variance_ratio_.sum():.0%} phương sai)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="Cụm")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_genre_heatmap(share, path, title, top_genres=12):
    cols = share.mean().sort_values(ascending=False).head(top_genres).index
    fig, ax = plt.subplots(figsize=(11, 0.9 * len(share) + 2))
    sns.heatmap(share[cols], annot=True, fmt=".0%", cmap="YlGnBu", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Cụm")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
