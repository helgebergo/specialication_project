import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import bus_model
import time
from numpy import random
from mpl_toolkits.mplot3d import Axes3D

plt.style.use(['seaborn-paper','master.mplstyle'])

def simulate_bus(params, n=20, plot_results=True):
    '''Simulates the bus model n times, and summarises the results. Can be used to
    plot if plot_results=True.'''
    results = {}
    nodes = {}
    total = {'peak':{}, 'not peak':{}}
    l = {}
    for peak in [True, False]:
        for i in range(n):
            params['peak'] = peak
            results[i], nodes[i], l[i] = bus_model.run(params)
        totals = {'passengers':[], 'stand':[], 'on':[], 'off':[], 
                  'current_inf':[], 'total_inf':[], 'newly_inf':[],
                  'means':{}, 'std':{}}
        for result in results.values():
            res = []
            on, off, passengers, stand, c_inf, t_inf, n_inf = [], [], [], [], [], [], []
            for stop in result.values():
                res.append(stop.values())
                on.append(stop['on'])
                off.append(stop['off'])
                passengers.append(stop['passengers'])
                stand.append(stop['stand'])
                c_inf.append(stop['current_inf'])
                t_inf.append(stop['total_inf'])
                n_inf.append(stop['newly_inf'])

            totals['on'].append(on)
            totals['off'].append(off)
            totals['passengers'].append(passengers)
            totals['stand'].append(stand)
            totals['current_inf'].append(c_inf)
            totals['total_inf'].append(t_inf)
            totals['newly_inf'].append(n_inf)
    
        for key in list(totals.keys())[0:-2]:
            totals['means'][key] = np.mean(totals[key], axis=0)
            totals['std'][key] = np.std(totals[key], axis=0)

        if peak:
            total['peak'] = totals
        else:
            total['not peak'] = totals
    
    if plot_results:
        plot_passenger_distributions(total, params, n)
        plot_disease_spread(total, params, n)


def plot_passenger_distributions(total, params, n):
    '''Plots a summary plot of the bus model, calculating the mean values
    from all model runs. '''
    stops = np.arange(1,params['num_stops']+1).tolist()
    
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(12, 8)
    
    for peak, col, label in zip(total.keys(), cycle, ['Peak hour', 'Non peak hour']):
        totals = total[peak]

        for key, ax in zip(totals.keys(), fig.axes):
            ax.plot(stops, totals['means'][key], label=label, color=col)
            ax.fill_between(stops, y1 = totals['means'][key] + totals['std'][key], 
                                y2=totals['means'][key] - totals['std'][key], alpha=0.2, color=col)
    
    fig.axes[0].set_ylabel('Number of passengers')
    fig.axes[1].set_ylabel('Number of standing passengers')
    fig.axes[2].set_ylabel('Passengers embarking')
    fig.axes[3].set_ylabel('Passengers disembarking')
    [ax.legend(frameon=False) and ax.set_xlabel('Stop number') and ax.set_ylim(-0.5) for ax in fig.axes]
    [ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True)) for ax in fig.axes]
    
    plt.tight_layout()
    
    plt.savefig('plots/sim_output_{}_p={}.png'.format(n,params['p_infected']), dpi=300)


def plot_disease_spread(total,params, n):
    '''Plots a summary plot of the disease spread, calculating the mean values
    from all model runs. '''
    stops = np.arange(1,params['num_stops']+1).tolist()
    
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(12, 4)
    max_values = []

    for peak, ax in zip(total.keys(), fig.axes):
        passengers = np.cumsum(total[peak]['means']['on'])
        newly_inf = (total[peak]['means']['newly_inf'])
        passengers_std = (total[peak]['std']['on'])
        newly_inf_std = (total[peak]['std']['newly_inf'])
        
        ax.plot(stops, passengers, label='Total passengers',color='darkcyan')
        ax.fill_between(stops, y1=passengers+passengers_std, 
                                y2=passengers-passengers_std, alpha=0.2,color='darkcyan')
        
        ax.plot(stops, newly_inf, label='Infected passengers', color='firebrick')
        ax.fill_between(stops, y1=newly_inf+newly_inf_std, 
                                y2=newly_inf-newly_inf_std, alpha=0.2,color='firebrick')
        ax.text(x=0.85*stops[-1],y=1.1*passengers[-1],
            s='{:1.0f} passengers'.format(passengers[-1]))
        
        if newly_inf[-1] == 0 or total[peak]['means']['total_inf'][0] == 0:
            R = 0
        else:
            R = newly_inf[-1]/total[peak]['means']['total_inf'][0]
        
        ax.text(x=0.85*stops[-1],y=1.05*(newly_inf[-1]+newly_inf_std[-1]),
            s='{:1.0f} infected\n$R$ = {:1.1f}'.format(np.round(newly_inf[-1]),R))

        ax.set_xlabel('Stop number')
        ax.set_ylabel('Number of passengers')
        
        max_values.append(1.2*passengers[-1])
    
    for ax in fig.axes:
        ax.legend(frameon=False,loc='upper left')
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax.set_ylim(-0.5,max(max_values))
    axs[0].set_title('Peak')
    axs[1].set_title('Non peak')

    plt.tight_layout()
    plt.savefig('plots/disease_output={}_p={}.png'.format(n,params['p_infected']), dpi=300)


def simulate_bus_v2(params, n=10):
    '''A simpler simulation program for used in the plots further down. '''
    results = {}
    nodes = {}
    df = np.empty((params['num_stops'],7,n))
    infected = []
    for i in range(n):
        results[i], nodes[i], df[:,:,i] = bus_model.run(params)
        infected.append(bus_model.get_newly_infected_fraction(nodes[i]['passengers']))

    return df, nodes, infected


def infected_boxplot():
    '''Creates a boxplot for different values of infection probability.'''
    prob_infected = [0,0.025,0.05,0.10,0.15,0.20,0.25]
    infected = {}
    for i, p_inf in enumerate(prob_infected,1):
        params = bus_model.set_parameters(p_infected=p_inf)
        _, _, infected[p_inf] = simulate_bus_v2(params,n=1000)
        progress(i,len(prob_infected))

    df = pd.DataFrame(infected)
    df.boxplot(grid=False,sym='.')

    plt.xlabel('Probability of infection')
    plt.ylabel('Fraction of infected passengers')
    plt.savefig('plots/boxplots/infected_boxplot.png', dpi=300)
    plt.close('all')


def stops_boxplot():
    '''Creates a boxplot for different number of stops.'''
    stops = [5,10,30,50,100,200]
    infected = {}
    for i,s in enumerate(stops,1):
        params = bus_model.set_parameters(num_stops=s)
        _, _, infected[s] = simulate_bus_v2(params,n=10)
        progress(i,len(stops))

    df = pd.DataFrame(infected)
    df.boxplot(grid=False,sym='.')

    plt.xlabel('Number of stops')
    plt.ylabel('Fraction of infected passengers')
    plt.savefig('plots/boxplots/stops_boxplot.png', dpi=300)
    plt.close('all')


def initial_infected_boxplot():
    '''Creates a boxplot for different values of initial infected.'''
    initial_infected = [0,0.2,0.4,0.6,0.8,1]
    infected = {}
    for i,inf in enumerate(initial_infected,1):
        params = bus_model.set_parameters(initial_infected=inf)
        _, _, infected[inf] = simulate_bus_v2(params,n=1000)
        progress(i,len(initial_infected))

    df = pd.DataFrame(infected)
    df.boxplot(grid=False,sym='.')

    plt.xlabel('Initial infected')
    plt.ylabel('Fraction of infected passengers')
    plt.savefig('plots/boxplots/initial_infected_boxplot.png', dpi=300)
    plt.close('all')


def lognormal_boxplot():
    '''Creates a boxplot for different values of the lognormal distribution.'''
    lognormal = [1,1.2,1.4,1.6,1.8,2]
    infected = {}
    for i,l in enumerate(lognormal,1):
        params = bus_model.set_parameters(lognorm=l)
        _, _, infected[l] = simulate_bus_v2(params,n=1000)
        progress(i,len(lognormal),len(lognormal))

    df = pd.DataFrame(infected)
    df.boxplot(grid=False,sym='.')
    
    plt.xlabel('Lognormal shape')
    plt.ylabel('Fraction of infected passengers')
    plt.savefig('plots/boxplots/lognormal_boxplot.png', dpi=300)
    plt.close('all')


def parameter_test_boxplot(parameter, values, n=100):
    '''Parameterised boxplot function.'''
    infected = {}
    for i, val in enumerate(values,1):
        params = bus_model.set_parameters()
        params[parameter] = val
        _, _, infected[val] = simulate_bus_v2(params, n)
        progress(i,len(values))

    df = pd.DataFrame(infected)
    return df


def make_boxplots(n=500):
    '''Creates a subplot with 4 boxplots in 1.'''
    parameters = ['initial_infected', 
                  'num_stops', 
                  'p_infected',
                  'trip_length']
    values = [[0,0.2,0.4,0.6,0.8,1],
              [10,20,30,50,100,200],
              [0,0.2,0.4,0.6,0.8,1],
              [2,4,6,8,10,15]]
    labels = ['Fraction of initially infected passengers', 
              'Number of stops', 
              'Probability of infection',
              'Average trip length']
    fig, axs = plt.subplots(2,2)
    fig.set_size_inches(12,8)

    for par, val, lab, axs in zip(parameters, values, labels, fig.axes):
        df = parameter_test_boxplot(par, val, n)
        df.boxplot(grid=False,sym='.',ax=axs)
        axs.set_xlabel(lab)
        axs.set_ylabel('Fraction of infected passengers')
    
    plt.tight_layout()
    plt.savefig('plots/boxplots/boxplots_all.png'.format(), dpi=100)


def calculate_grid(steps=20, simulations=20):
    '''Calculates a grid of z-values for the 3d plots'''
    infected = {}
    X = np.linspace(0,1,steps)
    Y = np.linspace(0,1,steps)
    print('{}x{}^2 = {} iterations'.format(simulations,steps,simulations*steps*steps))
    Z = np.zeros((steps,steps))

    params = bus_model.set_parameters()
    for i, x in enumerate(X):
        params['p_infected'] = x
        for j, y in enumerate(Y):
            params['initial_infected'] = y
            _, _, infected = simulate_bus_v2(params, simulations)  
            Z[j,i] = np.mean(infected)
        progress(i*steps,steps*steps)
    
    np.savez('grid',Z=Z,X=X,Y=Y)
    
    return X, Y, Z


def contour_plot():
    '''Creates a contour plot from the grid values '''
    try:
        npzfile = np.load('grid.npz')
        Z, X, Y = npzfile['Z'], npzfile['X'], npzfile['Y']
        print('loaded grids')
    except:
        print('calculates grids')
        X, Y, Z = calculate_grid()

    fig, ax = plt.subplots()
    fig.set_size_inches(12,8)

    cont = ax.contourf(X,Y,Z, alpha=0.8, cmap=plt.cm.viridis)
    cbar = plt.colorbar(cont, shrink=1)
    cbar.ax.set_ylabel('Fraction of infected passengers')

    ax.set_xlabel('Probability of infection')
    ax.set_ylabel('Initial infected')
    plt.tight_layout()
    plt.savefig('plots/surface_plots/contour_plot.png'.format(), dpi=300)


def surface_plot():
    '''Creates a 3D surface plot from the grid values '''
    try:
        npzfile = np.load('grid.npz')
        Z, X, Y = npzfile['Z'], npzfile['X'], npzfile['Y']
        print('loaded grids')
    except:
        print('calculates grids')
        Z, X, Y = calculate_grid()

    X, Y = np.meshgrid(X, Y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    fig.set_size_inches(12,8)

    surf = ax.plot_surface(Y,X,Z, cmap=plt.cm.viridis, antialiased=True) 
    cbar = fig.colorbar(surf, shrink=0.5)
    cbar.ax.set_ylabel('Infected passengers')
 
    ax.set_xlabel('Probability of infection')
    ax.set_ylabel('Initial infected')
    plt.tight_layout()
    plt.savefig('plots/surface_plots/surface_plot.png'.format(), dpi=300) 


def progress(iteration, total, bar_length=100):
    '''
    Generates a terminal progress bar when called in a loop.
    --> Adapted from https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a

    :param iteration: Integer, the current iteration.
    :param total: Integer, the number of total iterations.
    :param bar_length: Integer, character length of bar.
    :return: None
    '''
    percents = f'{100 * (iteration / float(total)):.2f}'
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = f'{"█" * filled_length}{"-" * (bar_length - filled_length)}'
    print(f'\rProgress: |{bar}| {percents}% complete ', end=''),
    if iteration == total:
        print("\n")


def main():
    params = bus_simulator.set_parameters(p_infected=1,lognorm_shape=1.4, 
                                          num_stops=100, num_seats=20,trip_length=6)
    simulate_bus(params,n=100,plot_results=True)
    
    make_boxplots(50)
    contour_plot()


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    duration = round(end-start,1)
    if duration > 100:
        print('Finished simulation in:',round(duration/60,1),'minutes.\n')
    else:
        print('Finished simulation in:',duration,'seconds.\n')
