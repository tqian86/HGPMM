#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys, csv, gzip, copy
import numpy as np
from collections import Counter, OrderedDict

_, samplefile = sys.argv

def reorder_by_freq(int_list):
    int_list = np.array(int_list)
    old_list = list(int_list)
    new_list = copy.deepcopy(int_list)
    #c = OrderedDict(sorted(Counter(int_list).items(), key = lambda t: (t[1], -1 * old_list.index(t[0])), reverse=True))
    c = OrderedDict(sorted(Counter(int_list).items(), key = lambda t: t[1], reverse=True))
    #c = Counter(int_list)
    new_i = 1
    for i in c.iterkeys():
        new_list[int_list==i] = new_i
        new_i += 1
    #print(c)
    #print('old:', int_list)
    #print('new:', new_list)
    return list(new_list)

def smallest_unused_label(int_labels):
    
    int_labels = set(int_labels)
    all_labels = set(xrange(1, max(int_labels) + 2))
    return min(all_labels - int_labels)


def reorder_by_comp(last_list, this_list):

    last_list = [int(_) for _ in last_list]
    this_list = [int(_) for _ in this_list]
    if last_list == this_list: return(this_list)

    count_dict = {}
    for i in xrange(len(last_list)):
        #if last_list[i] != this_list[i]: 
        try: count_dict[this_list[i]].append(last_list[i])
        except KeyError: count_dict[this_list[i]] = [last_list[i]]
    
    map_dict = {}
    used = []
    for key, values in count_dict.iteritems():
        for u in used:
            values = filter(lambda a: a != u, values)
        c = Counter(values)
        #print(key, c)
        try: 
            most_common = c.most_common(1)[0][0]
            map_dict[key] = most_common
            used.append(most_common)        
        except:
            pass

    for i in xrange(1, max(map_dict.values()) + 1):
        if i not in map_dict:
            map_dict[i] = smallest_unused_label(map_dict.values())
        
    #print(count_dict, map_dict)
    this_list = np.array(this_list)
    new_list = copy.deepcopy(this_list)
    for old, new in map_dict.iteritems():
        new_list[this_list==old] = new

    #print(new_list)
    return list(new_list)
    

#print(reorder_by_comp([1,1,1,2,2,2,3,3], [1,2,2,3,3,4,4,4]))
#sys.exit(0)

last = None
with gzip.open(samplefile) as csvfile:
    reader = csv.reader(csvfile)
    header = reader.next()
    print(*header, sep=',')
    for row in reader:
        if last is None: 
            print(*row, sep=',')
            last = copy.deepcopy(row[5:])
        else:
            new_list = reorder_by_comp(last, row[5:])
            print(*(row[:5] + new_list), sep=',')
            last = copy.deepcopy(row[5:])
