CSE425 Project – Variational Autoencoder (VAE) vs Autoencoder (AE) on MNIST
----------------------------------------------------------------------------

Overview:
This project compares a deterministic Autoencoder (AE) with a Variational Autoencoder (VAE) on the MNIST dataset.
The focus is on clustering performance, reconstruction quality, and uncertainty estimation.

Features:
- VAE with reparameterization trick (stochastic latent variables).
- AE baseline for deterministic comparison.
- Evaluation using Reconstruction MSE, Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), and Silhouette Score.
- Uncertainty quantification using posterior variance in latent dimensions.
- Results averaged across multiple seeds with statistical significance testing.

Requirements:
- Python 3.10 or later
- torch==2.2.0
- torchvision==0.17.0
- numpy==1.26.4
- matplotlib==3.8.3
- scikit-learn==1.4.1.post1
- scipy==1.12.0
- pandas==2.2.1

Install all dependencies using:
    pip install -r requirements.txt

How to Run:
1. Open the Jupyter notebook: CSE425_Project.ipynb
2. Run all cells in order.
3. The code will:
   - Train AE and VAE models
   - Save reconstructions and latent visualizations in the "outputs/" folder
   - Compute evaluation metrics (MSE, ARI, NMI, Silhouette)
   - Perform statistical tests between AE and VAE results
   - Generate an uncertainty analysis plot of latent posterior standard deviations

Outputs:
- Reconstruction images (original vs AE vs VAE)
- Latent clustering visualizations (PCA/TSNE)
- Uncertainty plot of posterior std per latent dimension
- Results table with mean ± std across seeds and statistical test p-values

