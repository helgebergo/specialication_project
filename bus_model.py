import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from numpy import random

mpl.style.use('seaborn-paper')


def initialize(params):
    '''Creates seats, doors and passengers for the simulation'''
    seats = create_nodes(params['num_seats'], params['num_stops'])
    doors = create_nodes(params['num_doors'], params['num_stops'])
    passengers = create_passengers(params)
    
    return seats, doors, passengers


def create_nodes(num_nodes, num_stops):
    '''Creates door and seat nodes, with the same parameters'''
    nodes = {}
    for i in range(1,num_nodes+1):
        nodes[i] = {'ID':i, 'occupied':0, 
                    'occupied_list':np.zeros(num_stops+1),
                    'persons':[], 'contaminated':False}
    
    return nodes


def create_passengers(params):
    '''Create passengers and initialise variables and lists after 
    the lognormal distribution.'''
    passenger_distribution = create_passenger_distribution(params)
    max_passengers = (params['num_seats']*4 + params['num_doors']*12) * params['num_stops']/8
    
    passengers = {}
    id = 1
    for stop, passengers_per_stop in enumerate(passenger_distribution, start=1):
        for t in range(passengers_per_stop):
            passengers[id] = {'ID':id, 'status':'S', 'node':None,
                            'present':False, 'standing':False,
                            'on':stop}
            trip_length = random.poisson(params['trip_length'], size=1)
            trip_length[trip_length < 1] = 1
            passengers[id]['off'] = min(stop+trip_length,params['num_stops'])
            id += 1

    return passengers


def create_passenger_distribution(params):
    '''Creates passengers following a lognormal distribution'''
    if params['peak']:
        log_params = params['lognorm']['peak'].values()
    else:
        log_params = params['lognorm']['off_peak'].values()
    
    passenger_distribution = stats.lognorm.rvs(*log_params,size=params['num_stops']-1).astype(int)

    passenger_distribution[passenger_distribution < 0] = 0
    passenger_distribution[passenger_distribution > params['passenger_limit']] = params['passenger_limit']
    
    return passenger_distribution


def embarking(passengers, seats, current_stop, doors):
    '''Loops through all passengers with the current stop as first, and uses 
    find_place() to place them. Returns number of passengers embarking.'''
    on = 0
    for passenger in passengers.values():
        if passenger['on'] == current_stop:
            find_place(doors, seats, passenger, current_stop)
            on += 1
    return on


def disembarking(passengers, seats, current_stop):
    '''Loops through all passengers with this stop as last, and uses 
    remove_passenger_from_node(). Returns number of passengers disembarking.'''
    off = 0
    for passenger in passengers.values():
        if passenger['off'] == current_stop:
            remove_passenger_from_node(passenger['node'], passenger, current_stop)
            passenger['standing'] = False
            off += 1
    return off


def find_place(doors, seats, passenger, current_stop):
    '''Loops through n=3 nearest seats for a passenger, and seats them based on probabilities.
    If not, let them stand by a door.'''
    if passenger['standing']:
        door = passenger['node']['ID']
    else:
        door = random.randint(1, len(doors)+1)
    num_seats = len(seats)
    seats_per_door = int(num_seats/len(doors))
    nearest_seats = list(range(num_seats))[seats_per_door*door:seats_per_door*(door+1)]
    # nearest_seats = list(range(1,len(seats)+1))  # uncomment if all seats should be available
    random.shuffle(nearest_seats)
    seated = False
    for occ, prob in zip([0,1,2,3],[1,1,0.5,0.05]):
        for i in nearest_seats:
            if seats[i]['occupied'] == occ and random.random() < prob and not seated:
                assign_passenger_to_node(seats[i], passenger, current_stop)
                passenger['standing'] = False
                seated = True
                return
    
    if not seated:
        assign_passenger_to_node(doors[door], passenger, current_stop)
        passenger['standing'] = True
        

def assign_passenger_to_node(node, passenger, current_stop):
    '''Bookkeeping for assignment of a passenger to a door or seat.'''
    node['persons'].append(passenger)
    node['occupied'] += 1
    node['occupied_list'][current_stop] += 1
    passenger['node'] = node
    passenger['present'] = True


def remove_passenger_from_node(node, passenger, current_stop):
    '''Bookkeeping for removing a passengers from a node.'''
    node['persons'].remove(passenger)
    node['occupied'] -= 1
    passenger['node'] = None
    passenger['present'] = False
    passenger['standing'] = False


def switchplaces(passengers, seats, stop, doors):
    '''For all standing passengers on the bus, check to see if there are any 
    available seats nearby, and then seat them there using the find_place function'''
    seat_switch_percentage = 0.33
    for passenger in passengers.values():
        if passenger['standing'] and random.random() > seat_switch_percentage:
            find_place(doors, seats, passenger, stop)


def check_spread(seats, doors, passengers, stop, params):
    '''For all infected passengers currently on the bus, this function uses a 
    binomial draw to infect the nearest nodes'''
    for passenger in passengers.values():
        if passenger['status'] == 'I' and passenger['present'] and passenger['ancestor'] is None:
            node = passenger['node']
            neighbors = node['persons']
            try:
                k = random.binomial(len(neighbors), params['p_infected'])
            except ValueError:
                return
            inf_neighbors = random.choice(neighbors, k, replace=False)
            for nb in inf_neighbors:
                infect_node(nb, passenger, stop)
            

def infect_node(node, ancestor, stop):
    '''Sets status to 'I', and logs where and from whom the node was infected.'''
    if node['status'] == 'I':
        return
    node['status'] = 'I'
    node['ancestor'] = ancestor
    node['log'] = stop


def infect_random_passengers(passengers, params, k=None):
    '''Infects k random passengers from a binomial draw. Returns the amount'''
    if not k:
        k = random.binomial(len(passengers), params['initial_infected'])
    infected = random.choice(list(passengers), k, replace=False)
    for i in infected:
        infect_node(passengers[i], ancestor=None, stop=None)
    return k


def get_seated_count(passengers, seats):
    seated = 0
    for seat in seats:
        seated += seats[seat]['occupied']
    return seated


def get_seated_passengers(passengers, seats):
    seated = []
    for seat in seats:
        seated.append(seats[seat]['occupied'])
    return seated


def get_infected(passengers):
    '''Returns all infected nodes. For use after the simulation is done. Does 
    not take into account if passengers are present or not'''
    infected = [p for p in passengers.values() if p['status'] == 'I']
    return infected


def get_initial_infected(passengers):
    '''Returns all initially infected nodes. Does not take into account if passengers are present or not'''
    initial_infected = [p for p in passengers.values() if p['status'] == 'I' and p['ancestor'] is None]
    return initial_infected


def get_standing_passengers(passengers):
    standing = [p for p in passengers.values() if p['standing'] and p['present']]
    return len(standing)


def get_current_infected(passengers, stop):
    present_infected = len([p for p in passengers.values() if p['status']=='I' and p['present']])
    total_infected = len([p for p in passengers.values() if p['status']=='I'])
    newly_infected = total_infected - len(get_initial_infected(passengers))
    return present_infected, total_infected, newly_infected


def get_newly_infected(passengers):
    total_infected = get_infected(passengers)
    initial_infected = get_initial_infected(passengers)
    return len(total_infected) - len(initial_infected)


def get_newly_infected_fraction(passengers):
    newly_infected = get_newly_infected(passengers)
    total_passengers = len(passengers)
    try:
        return newly_infected/total_passengers
    except ZeroDivisionError:
        return 0


def simulate_bus(seats, doors, passengers, params):
    results = {}
    passenger_count = 0
    seated_passengers = []
    l = []
    for stop in range(1, params['num_stops']+1):
        off = disembarking(passengers, seats, stop)
        switchplaces(passengers, seats, stop, doors)
        on = embarking(passengers, seats, stop, doors)
        check_spread(seats, doors, passengers, stop, params)

        passenger_count = passenger_count + on - off
        standing_passengers = get_standing_passengers(passengers)
        infected = get_current_infected(passengers, stop)
        results[stop] = {'off':off,'on':on,'stand':standing_passengers,
                        'passengers':passenger_count, 
                        'current_inf':infected[0], 
                        'total_inf':infected[1],
                        'newly_inf':infected[2]}
        l.append([off,on,standing_passengers,passenger_count,*infected])
    df = pd.DataFrame(l, columns=['off','on','standing','passengers','current_inf','total_inf','newly_inf'])
    return results, df


def set_parameters(**kwds):
    '''New version with keywords instead of simpler method. 
    kwds: num_stops, peak, num_seats, num_doors, initial_infected,
    p_infected, lognorm, dist, trip_length, passenger_limit'''
    p = {}
    p['num_stops'] = kwds.get('num_stops', 30)
    p['peak'] = kwds.get('peak', True)
    p['num_seats'] = kwds.get('num_seats', 12)
    p['num_doors'] = kwds.get('num_doors', 4)

    p['initial_infected'] = kwds.get('initial_infected', 0.05)
    p['p_infected'] = kwds.get('p_infected', 0.01)
    
    p['lognorm'] = {'peak':     {'shape': 0.76, 'loc': -1.00, 'scale': 2.72},
                     'off_peak':{'shape': 0.63, 'loc': -1.26, 'scale': 2.72}}
    p['lognorm']['peak']['shape'] = kwds.get('lognorm_shape', 0.76)
    p['lognorm']['peak']['loc'] = kwds.get('lognorm_loc', -1.00)
    p['lognorm']['peak']['scale'] = kwds.get('lognorm_scale', 2.72)

    p['trip_length'] = kwds.get('trip_length', 4)
    p['passenger_limit'] = kwds.get('passenger_limit', 58)

    return p


def print_summary(nodes):
    passengers = len(nodes['passengers'])
    print('{:3} passengers'.format(passengers))

    infected = len(get_infected(nodes['passengers']))
    initial_infected = len(get_initial_infected(nodes['passengers']))
    newly_infected = infected - initial_infected

    print('{:3} initially infected'.format(initial_infected))
    print('{:3} newly infected'.format(newly_infected))
    print()
    print('{:.1%} initially infected'.format(initial_infected/passengers))
    print('{:.1%} newly infected'.format(newly_infected/passengers))
    try:
        print('R = {:1.1}'.format(newly_infected/initial_infected))
    except ZeroDivisionError:
        return
    print()


def run(params):
    seats, doors, passengers = initialize(params)
    infect_random_passengers(passengers,params)
    results, df = simulate_bus(seats, doors, passengers, params)
    
    nodes = {'seats':seats, 'doors':doors, 'passengers':passengers}

    return results, nodes, df


def main():
    params = set_parameters()

    results, nodes, df = run(params)
    print_summary(nodes)
    
    return results


if __name__ == '__main__':
    main()
