from math import ceil
from .gtfs_networks import *
import geopandas as gpd
import pickle
from joblib import Parallel, delayed
from .accessibility import *
import numpy as np
import random

def assign_buses(lines,bus_gene,one_at_least=False,mode=None):
    if one_at_least:
        if mode is None:
            buses={L.graph['route_id']:1 for L in lines}
        else:
            buses={L.graph['route_id']:1 for L in lines if L.graph['mode']==mode}
    else:
        if mode is None:
            buses={L.graph['route_id']:0 for L in lines}
        else:
            buses={L.graph['route_id']:0 for L in lines if L.graph['mode']==mode}
    for g in bus_gene:
        buses[g]+=1
    for L in lines:
        if mode is not None:
            if L.graph['mode'] != mode:
                continue
        if buses[L.graph['route_id']]!=0:
            L.graph['headway']=ceil(L.graph['cycle_time']/buses[L.graph['route_id']]/300)*300
        else:
            L.graph['headway']=1e30
    return buses

def get_buses_per_route(lines,mode=None):
    buses={}
    for L in lines:
        if mode is not None:
            if L.graph['mode'] != mode:
                continue
        buses[L.graph['route_id']]=ceil(L.graph['cycle_time']/L.graph['headway'])
    return buses

def line_in_area(L,G):
    for _,node in L.nodes(data='attached'):
        if node in G.nodes:
            break
    else:
        return False
    return True

def line_strictly_in_area(L,G):
    for _,node in L.nodes(data='attached'):
        if node not in G.nodes:
            return False
    else:
        return True

def shutdown(buses,b):
    if b in buses:
        buses[b]=0
    else:
        raise KeyError(f'{b} not in buses')
        
def reduce_frequency(buses,b,by=2):
    if b in buses:
        buses[b]=round(buses[b]/by)
    else:
        raise KeyError(f'{b} not in buses')
        
def assign_buses(lines,bus_gene,one_at_least=False,mode=None,
                failed=[],fail_process=shutdown,fail_kws={}):
    if one_at_least:
        if mode is None:
            buses={L.graph['route_id']:1 for L in lines}
        else:
            buses={L.graph['route_id']:1 for L in lines if L.graph['mode']==mode}
    else:
        if mode is None:
            buses={L.graph['route_id']:0 for L in lines}
        else:
            buses={L.graph['route_id']:0 for L in lines if L.graph['mode']==mode}
    
    for g in bus_gene:
        buses[g]+=1
    for b in buses:
        if b in failed:
            fail_process(buses,b,**fail_kws)
    for L in lines:
        if mode is not None:
            if L.graph['mode'] != mode:
                continue
        if buses[L.graph['route_id']]!=0:
            L.graph['headway']=ceil(L.graph['cycle_time']/buses[L.graph['route_id']]/300)*300
        else:
            L.graph['headway']=1e30
    return buses    

def performance_check(s,lines,G,gdf,func_kws=func_kws,one_at_least=False,
                     failed=[],fail_process=shutdown,fail_kws={}):
    lines = [L.copy() for L in lines]
    bs=assign_buses(lines,s,one_at_least, failed=failed,
                    fail_process=fail_process)
    Gtransit=G.copy()
    for L in lines:
        add_bus_line(Gtransit,L,L.graph['name'],headway=L.graph['headway'])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acc=calc_zone_accessibility(gdf,Gtransit,opportunities_column='jobs',
                                    func=soft_threshold,func_kws=func_kws,
                                    weight='time_sec', competition=False,
                                    random_seed=1,k=3,node_subset=G.nodes)
    if one_at_least:
            w=(1-0.1*len([n for n in bs.values() if n==0]))
            return (np.average(list(acc.values()),weights=gdf['population']))*w
    else:
        return np.average(list(acc.values()),weights=gdf['population'])
        
def selection(pop, scores, k=3):
    selection_ix = random.randint(0,len(pop)-1)
    for ix in np.random.randint(0, len(pop), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] > scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]

def crossover(gene1,gene2,rate):
    if random.random()<rate:
        c1,c2=[],[]
        pt=random.randint(1, len(gene1)-2)
        c1 += gene1[:pt] + gene2[pt:]
        c2 += gene2[:pt] + gene1[pt:]
        return [c1, c2]
    else:
        return [gene1,gene2]
    
def mutation(a,possible,r_mut=.01):
    a=a.copy()
    for i in range(len(a)):
        # check for a mutation
        if random.random() < r_mut:
            a[i] = random.choice(possible)
    return a
import time

def GA(func,G,lines,gdf,bus_num,population=32,sufix='',one_at_least=False,
       seed=None,crossover_rate=.9,generations=100,r_mut=.01,func_kws={'t':30},
       take=2,first_guesses=None,save=True,saveeach=1,parallel=True,workers=-1,
       init_density=.3,fail_rate=0,fail_process=shutdown,fail_kws={}):
    gen=1
    baseline_value=performance_check([],lines,G,zones,func_kws=func_args[0])
    if one_at_least:
        bus_num=bus_num-len(lines)
        if bus_num<=0:
            raise ValueError(f'Not enough buses for "one_at_least". Have {len(lines)} routes for {bus_num+len(lines)} buses')
    possible = [L.graph['route_id'] for L in lines]
    #start with random population
    random.seed(seed)
    if first_guesses is None:
        first_guesses=[]
    sets=first_guesses
    sets+= [[random.choice(possible) for n in range(bus_num)] 
            for n in range(population-len(first_guesses))]
    
    while True:
        t=time.time()
        try:
            failed=[]
            for L in lines:
                if random.random()<fail_rate:
                    failed.append(L.graph['route_id'])
            #parallel:
            if parallel:
                scores = Parallel(n_jobs=workers)(delayed(func)(s,lines=lines,G=G,gdf=gdf,
                                                                one_at_least=one_at_least,
                                                                func_kws=func_kws,
                                                                failed=failed,
                                                                fail_process=fail_process,
                                                                fail_kws=fail_kws) 
                                                  for s in sets)
                scores = [s-baseline_value for s in list(scores)]
            #singlethread
            else:
                scores = [func(s,lines=lines,G=G,gdf=gdf,
                          one_at_least=one_at_least,
                          func_kws=func_kws) for s in sets]
        except KeyboardInterrupt:
            return sets,scores
        top_choices = [selection(sets,scores) for _ in range(population)]
        if gen>generations:break
        #do crossover
        set_dict = {n:s for n,s in zip(range(len(sets)),sets)}
        sets = [set_dict[n] for _,n in sorted(zip(scores,range(len(sets))))[-take:]]
        
        for i0 in range(take, population, 2):
            # get selected parents in pairs
            p1, p2 = top_choices[i0], top_choices[i0+1]
            # crossover and mutation
            for c in crossover(p1, p2, crossover_rate):
                # mutation
                c=mutation(c,possible, r_mut)
                # store for next generation
                #shuffle
                sets.append(random.sample(c,k=len(c)))
        print(f'gen {gen-1}\ttop: {max(scores):.00f}\tpop: {len(scores)}\titer time: {time.time()-t:.00f}s')
        if (gen-1)%saveeach==0:
            pickle.dump((sets,scores),open(f'results/{sufix}_gen{gen-1}.dat','wb'))
        gen+=1
    print(f'gen {gen-1} top: {max(scores)}')
    return sets,scores

def edge_statistics(edge_types,route_dict,time_dict,path_mat,
                    filter_areas=None):
    stats = {'trip_id':[],'etype':[],'route':[],'total_travel_time':[],
             'opportunities':[],'population':[],'time_on_edge':[],'Betweenness_Acc':[]}
    tid=0
    for w,z,t,path,opp,pop,dist_decay in path_mat:
        if filter_areas is None or z in filter_areas:
            for e in path:
                etype = edge_types[e]
                if etype is None:
                    etype='Walk'
                rtype = route_dict[e]
                if rtype is None:
                    rtype='Walk'
                stats['trip_id'].append(tid)
                stats['etype'].append(etype)
                stats['route'].append(rtype)
                stats['total_travel_time'].append(t)
                stats['time_on_edge'].append(time_dict[e])
                stats['opportunities'].append(opp)
                stats['population'].append(pop)
                stats['Betweenness_Acc'].append(w)
            tid+=1
    return stats