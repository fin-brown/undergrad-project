#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 14:34:22 2018

@author: finbrown1
"""

from bisect import bisect_left

DISTANCES = [100, 200, 400, 800, 1500, 1609, 5000, 10000, 21094, 42194]
DISTANCES_EXACT = [100., 200., 400., 800., 1500., 1609.3440000000001, 
                   5000., 10000., 21094.490000000002, 42194.989999999998]

def index_nearest(x, events):
    
    """ Given an event and a series of events, returns the index position
        of the nearest event. Computationally fast implementation. 
    """
    
    index, values = events.index, events.values
    
    pos = bisect_left(values, x)
    try:
        if pos == 0:
            return index[0]
    
        if pos == len(events):
            return index[-1]
        
    except IndexError as err:
        raise Exception(("{}: There are no events. This probably means the train"
                         " set has no records from the athlete.").format(err))
    
    before, after  = values[pos - 1:pos + 1]
    
    event = after if after - x < x - before else before
    
    return index[values==event][0]


def event_to_distance(events):
    
    return [DISTANCES_EXACT[event-1] for event in events]
    

def int_distance_to_event(distances):
    
    return [DISTANCES.index(distance) + 1 for distance in distances]


