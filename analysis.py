import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats

plt.style.use(['seaborn-paper','master.mplstyle'])


def read_txt_files():
    routes = {}
    df = pd.DataFrame()
    for r in ['2n', '2s','22n','22s']:
        route = {}
        for t in ['pass','stand', 'on', 'off', 'sit']:
            name = 'bus_data/r{}_{}.txt'.format(r, t)
            route[t] = pd.read_table(name, decimal=',')
        routes[r] = route
    
    data = pd.DataFrame.from_dict(routes)
    
    return routes


def read_txt_files_route2():
    data = {}
    df = pd.DataFrame()
    for t in ['pass','stand', 'on', 'off']:
        name = 'bus_data/route2{}.txt'.format(t)
        data[t] = pd.read_table(name, decimal=',')
    
    return data


def plot_bus_data(data, peak_hours=[7,8,9]):
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axs = plt.subplots(2,2)
    fig.set_size_inches(12,8)
    log_params = {}
    
    for ax, plot in zip(fig.axes, ['pass', 'stand', 'on', 'off']):
        data = data[plot]
        times = data.keys()[2:] 
        stops = data['stopp nr']

        peak = [data[time] for time in times if int(time.split(':')[0]) in peak_hours]
        npeak = [data[time] for time in times if int(time.split(':')[0]) not in peak_hours]
        
        rush_m = np.nanmean(peak, axis=0)
        rush_std = np.nanstd(peak, axis=0)
        n_rush_m = np.nanmean(npeak, axis=0)
        n_rush_std = np.nanstd(npeak, axis=0)

        ax.plot(stops, rush_m, label='Peak hour', linewidth=1, color=cycle[0])
        ax.fill_between(stops, y1 = rush_m + rush_std, y2=rush_m-rush_std, alpha=0.2, color=cycle[0])
        ax.plot(stops, n_rush_m, label='Non peak hour', linewidth=1, color=cycle[1])
        ax.fill_between(stops, y1 = n_rush_m + n_rush_std, y2=n_rush_m - n_rush_std, alpha=0.2, color=cycle[1])

        if plot == 'on':
            for df, label in zip([peak, npeak],['Peak','NPeak']):
                df = np.array(df)
                df = df[~np.isnan(df)]
                params = stats.lognorm.fit(df,fscale=1.0,loc=0)
                log_params[label] = params

    fig.axes[0].set_ylabel('Number of passengers')
    fig.axes[1].set_ylabel('Number of standing passengers')
    fig.axes[2].set_ylabel('Passengers embarking')
    fig.axes[3].set_ylabel('Passengers disembarking')
    [ax.legend(frameon=False) and ax.set_xlabel('Stop number') and ax.set_ylim(-1) for ax in fig.axes]
    [ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True)) for ax in fig.axes]
        
    plt.tight_layout()
    plt.savefig('plots/route2.png'.format(), dpi=300)
    
    return log_params


def test_normality(routes):
    alpha = 0.05
    message = ''
    for route_key in routes.keys():
        for route_type in routes[route_key].keys():
            onboarding = []
            data = routes[route_key][route_type]
            for key in list(data.keys())[2:-2]:
                onboarding.append(data[key])
                stat, p = stats.normaltest(data[key])
                if p > alpha:
                    message += '\n{}\t{}\t{}\tNormal:{}'.format(route_key,route_type,key,p>alpha)

                fig, ax = plt.subplots()
                ax.hist(data[key])
                ax.set_title('{} {}\nnormal: {}'.format(route_key,key,p>alpha))
                fig.tight_layout()
                # plt.savefig('plots/histograms/hist_{}-{}-{}'.format(route_type,route_key,key),dpi=300)
            
            # fig, ax = plt.subplots()
            # ax.hist(onboarding)
            # fig.tight_layout()
            # plt.savefig('plots/histograms/histogram_{}-{}'.format(route_key,route_type),dpi=200)

    print(message)


def test_distributions(distribution='lognorm',rush_hours=[7,8,9]):
    dist = getattr(stats, distribution)
    df = pd.read_table('bus_data/r2_on_all.txt')
    peak = pd.DataFrame([df[t] for t in df.keys()[1:] if int(t.split(':')[0]) in rush_hours]).values.flatten()
    peak = peak[~np.isnan(peak)]

    non_peak = pd.DataFrame([df[t] for t in df.keys()[1:] if int(t.split(':')[0]) not in rush_hours]).values.flatten()
    non_peak = non_peak[~np.isnan(non_peak)]
    
    fig, axs = plt.subplots(1,2)
    fig.set_size_inches(12,4)
    size = 100000
    x = np.linspace(0,30,size)
    bins = 31

    for d, ax in zip([peak, non_peak], fig.axes):
        ax.hist(d,bins=bins,range=(0,30),density=True,label='Empirical',alpha=0.5,color='firebrick')
        params = dist.fit(d,loc=1,fscale=np.exp(1))

        print('shape: {:2.2f}, loc: {:1.2f}, scale: {:1.2f}'.format(*params))
        print('\u03BC: {:2.1f}, \u03C3: {:1.2f}'.format(np.log(params[2]),params[0]))
        
        ax.plot(x, dist.pdf(x,*params),color='black',label='PDF, \u03BC={:2.1f}, \u03C3={:2.2f}'.format(np.log(params[2]),params[0]))
        y = dist.rvs(*params, size=size)
        y[y<0] = 0
        ax.hist(y,bins=bins,range=(0,30),density=True,label='Generated',alpha=0.5)
        ax.legend(frameon=False)
        ax.set_xlim(-0.5,15)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax.set_xlabel('$x$')
        ax.set_ylabel('$P(x)$')

    axs[0].set_title('Peak')
    axs[1].set_title('Non peak')
    fig.tight_layout()
    plt.savefig('plots/distributions/distribution_{}.png'.format(dist.name), dpi=300)


def main():
    routes = read_txt_files_route2()
    plot_bus_data(routes)


if __name__ == "__main__":
    main()
