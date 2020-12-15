import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

plt.style.use(['seaborn-paper','master.mplstyle'])
color_map = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=3), cmap=mpl.cm.viridis)
color_map.set_array([])


def plot_lognormal_distribution():
    x = np.linspace(0,18,1000)
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(8, 4)
    
    for (i, mu), sigma in zip(enumerate([1.0,1.5,2.0]),[1.5,1.0,0.5]):
        c = color_map.to_rgba(i)
        mean = np.exp(mu + (sigma**2)/2)

        ax.plot(x, stats.lognorm.pdf(x, sigma, scale=np.exp(mu)), linestyle='-', color=c)
        ax.fill_between(x, y1=0, y2=stats.lognorm.pdf(x, sigma, scale=np.exp(mu)),color=c, alpha=0.2)
        ax.vlines(mean, 0, stats.lognorm.pdf(mean, sigma, scale=np.exp(mu)), linestyle='--', color=c,
                                    label='$\mu={}$, $\sigma={}$'.format(mu,sigma))
    
    ax.set_xlabel("$x$")
    ax.set_ylabel("$P(x)$")
    
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('plots/statistics/lognormal_distribution.png', dpi=300)


def plot_normal():
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 4)
    x = np.linspace(-1,24,1000)

    for mu, var in zip([2,8,16],[1,2,3]):
        c = color_map.to_rgba(var-1)
        ax.plot(x, stats.norm.pdf(x,mu,var), label='$\mu={}, \sigma = {}$'.format(mu, var), color=c)
        ax.fill_between(x, y1=0, y2=stats.norm.pdf(x,mu,var), color=c,alpha=0.2)
        ax.vlines(mu, 0, stats.norm.pdf(mu,mu,var), linestyle='--', color=c)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$P(x)$')
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('plots/statistics/normal_distribution.png', dpi=300)


def plot_exponential_distribution():
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(8, 4)
    x = np.linspace(0,15,1000)
    
    for i, lambda_ in enumerate([1,0.5,0.25]):
        c = color_map.to_rgba(i)
        mean = 1/lambda_
        
        ax.plot(x, stats.expon.pdf(x, loc=0, scale=1/lambda_), linestyle='-', color=c, label='$\lambda={}$'.format(lambda_))
        ax.fill_between(x, y1=0, y2=stats.expon.pdf(x, loc=0, scale=1/lambda_), color=c,alpha=0.2)
        ax.vlines(mean, 0, stats.expon.pdf(mean, loc=0, scale=1/lambda_), linestyle='--', color=c)
    
    ax.set_xlabel("$x$")
    ax.set_ylabel("$P(x)$")

    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('plots/statistics/exponential_distribution.png', dpi=300)


def plot_binomial_distribution():
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8,4)

    for (i, p), n in zip(enumerate([0.5,0.5,0.8]),[4,16,20]):
        x = np.arange(stats.binom.ppf(0.001, n, p),stats.binom.ppf(0.999, n, p))
        c = color_map.to_rgba(i)
        ax.bar(x, stats.binom.pmf(x, n, p), color=c, linewidth=1, alpha=0.5, label='$p={}$, $n={}$'.format(p,n))
    
    ax.set_xlabel('$x$')
    ax.set_ylabel('$P(x)$')
    ax.legend(frameon=False)

    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    plt.savefig('plots/statistics/binomial_distribution.png', dpi=300)


def plot_poisson_distribution():
    '''Plots a poisson distribution. '''
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8,4)

    for i, lambda_ in enumerate([2,8,16]):
        c = color_map.to_rgba(i)
        x = np.arange(stats.poisson.ppf(0.01, lambda_),stats.poisson.ppf(0.99, lambda_))
        ax.bar(x, stats.poisson.pmf(x, lambda_), color=c, linewidth=1, alpha=0.5, label='$\lambda={}$'.format(lambda_))

    ax.set_xlabel('$x$')
    ax.set_ylabel('$P(x)$')
    ax.legend(frameon=False)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig('plots/statistics/poisson_distribution.png', dpi=300)


def plot_powerlaw_distribution():
    '''Plots a powerlaw distribution. '''
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8,4)
    x = np.linspace(0,1,1000)

    for i, a in enumerate([2,3,4]):
        c = color_map.to_rgba(i)
        ax.plot(x, stats.powerlaw.pdf(x, a), color=c, label='$\lambda={}$'.format(a))
        ax.fill_between(x, y1=0, y2=stats.powerlaw.pdf(x, a), color=c,alpha=0.2)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$P(x)$')
    ax.legend(frameon=False)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig('plots/statistics/powerlaw_distribution.png', dpi=300)


def plot_poisson():
    '''Plots a poisson distribution for the methods chapter. '''
    size = 10000000

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8,4)
    for i, mu in enumerate([1,4,7]):
        c = color_map.to_rgba(i)
        plt.hist(stats.poisson.rvs(mu, size=size),density=True,alpha=0.3, color=c)
        plt.vlines(mu, 0, stats.poisson.pmf(mu,mu),label='$\mu={}$'.format(mu),linestyle='--', color=c)
    
    plt.xlabel('$x$')
    plt.xlim(-.5,16)
    plt.ylabel('$P(x)$')
    ax.legend(frameon=False)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig('plots/statistics/poisson.png', dpi=300)


def main():
    plot_binomial_distribution()
    plot_poisson_distribution()
    plot_normal()

main()
