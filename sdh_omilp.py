# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 11:04:37 2015

@author: s1370831

sdh_omilp.py

SDH Optimisation 1:
- Given: Synthesis & Design
- Determine: Operational
- S.T.: Meeting the demand
- In order to: minimize operational cost (i.e. min boiler input / max solar
    fraction)
    
Equipment modelling:
- Overall: Energy stream-only, Temperature is not included.
- Solar collector: Constant efficiency
- Storage: Capacity model
- Boiler: Constant efficiency
- Control: Original DLSC control assumption (see Quintana & Kummert, 2014)
"""

from __future__ import division
from pyomo.environ import *

import numpy as np
import pandas as pd

import timeit

import matplotlib.pyplot as plt

import csv

def read_excel(filename):
    """
    Read Excel input file which contains equipment data and connectivity
    matrix.
    """
    with pd.ExcelFile(filename) as xls:
        conversion = xls.parse(
            'Conversion',
            index_col=['Equipment'])
        storage = xls.parse(
            'Storage',
            index_col=['Equipment'])
        demand = xls.parse(
            'Demand',
            index_col=['Timestep'])
        weather = xls.parse(
            'Weather',
            index_col=['Timestep'])
        sts_charge_required = xls.parse(
            'STS_Charge_Required',
            index_col=['Time'])
            
    data = {
        'conversion' : conversion,
        'storage' : storage,
        'demand' : demand,
        'weather' : weather,
        'sts_charge_required' : sts_charge_required}
        
    return data

def create_model(data):
    
    m = ConcreteModel(name="SDHMILP1")
    
    m.timesteps = data['demand'].index.tolist()
    
    m.conversion = data['conversion']
    m.storage = data['storage']
    m.demand = data['demand']
    m.weather = data['weather']
    m.sts_charge_required = data['sts_charge_required']
    
    # Sets
    # ====
    
    # generate ordered time step sets
    m.t = Set(
        initialize=m.timesteps,
        ordered=True,
        doc='Set of timesteps')
        
    # energy conversion equipment (e.g. boiler, chp, ...)
    m.con = Set(
        initialize=m.conversion.index.get_level_values('Equipment').unique(),
        ordered=True,
        doc='Set of conversion equipment')
        
    m.sto = Set(
        initialize=m.storage.index.get_level_values('Equipment').unique(),
        ordered=True,
        doc='Set of storage equipment')
    
    # Parameters
    # ==========
    
    # ambient temperature
#    m.temp_amb = Param(
#        m.timesteps,
#        initialize=m.weather.loc[t]['Temperature'] for t in m.timesteps,
#        doc='Set of ambient temperature')
    
    def dlsp_init(m, t):
        if m.weather.loc[t]['Temperature'] <= -40:
            return 55.0
        
        elif -40 < m.weather.loc[t]['Temperature'] < -2.5:
            return (-0.48 * m.weather.loc[t]['Temperature'] + 35.8)
        
        else:
            return 37.0
            
    m.dlsp = Param(
        m.t,
        initialize=dlsp_init,
        doc='Drake Landing Set Point')
        
    # Charge required lookup rule
    def ch_req_init(m, t):
        if m.dlsp[t] <= 38:
            return m.sts_charge_required.loc[m.demand.loc[t]['Hour']]['DLSP <= 38']
                    
        elif 38 < m.dlsp[t] < 45:
            return m.sts_charge_required.loc[m.demand.loc[t]['Hour']]['38 < DLSP < 45']
                    
        else:
            return m.sts_charge_required.loc[m.demand.loc[t]['Hour']]['DLSP >= 45']
        
    m.ch_req = Param(
        m.t,
        initialize=ch_req_init,
        doc='STS % Charge required')
        
    # Solar collector variables
    # Solar collector output constraint rule
    def sco_init(m, t):
        return (m.conversion.loc['SCO']['Efficiency'] *
                m.conversion.loc['SCO']['Size'] *
                m.weather.loc[t]['Global horizontal irradiation'])
                
    m.q_sco = Param(
        m.t,
        initialize=sco_init,
        doc='Solar collector output')
    
    # Variables
    # =========
   
    m.q_sco_hx1 = Var(
        m.t,
        within=NonNegativeReals,
        doc='HX1 energy from solar')
    
    # Boiler variables
    m.q_boi = Var(
        m.t,
        within=NonNegativeReals,
        doc='Boiler output')    
    
    # Storage variables
        
    m.q_ch = Var(
        m.t, m.sto,
        within=NonNegativeReals,
        doc='Charging rate')
        
    m.q_dch = Var(
        m.t, m.sto,
        within=NonNegativeReals,
        doc='Discharging rate')
        
    m.q_sto = Var(
        m.t, m.sto,
        within=NonNegativeReals,
        doc='Stored energy at t')
        
    m.percent_charge = Var(
        m.t, m.sto,
        within=NonNegativeReals,
        doc='Ratio between stored and maximum capacity')
        
    # Binary indicator variables
    
    # LTS control indicators
    m.delta_ch = Var(
        m.t,
        within=Binary,
        doc='Indicator of LTS charge')
        
    m.delta_dch = Var(
        m.t,
        within=Binary,
        doc='Indicator of LTS discharge')
 
    m.y = Var(
        m.t,
        within=Binary,
        doc='Indicator of STS-LTS IF-THEN control')
        
    # Dumping control indicators
#    m.delta_sts = Var(
#        m.t,
#        within=Binary,
#        doc='Indicator if STS is full')
#        
#    m.delta_lts = Var(
#        m.t,
#        within=Binary,
#        doc='Indicator if LTS is full')
#        
#    m.delta_extra = Var(
#        m.t,
#        within=Binary,
#        doc='delta_extra == delta_sts * delta_lts')
#    
#    m.delta_sts_ch = Var(
#        m.t,
#        within=Binary,
#        doc='charge sts ON/OFF')
    # Constraints
    # ===========
    # (constraints rules are defined separately)
        
    # Solar collector output constraint
    m.c_sco_hx1 = Constraint(
        m.t,
        rule=c_sco_hx1_rule,
        doc='q_sco_hx1 <= q_sco')
        
    # STS charge constraint
    m.c_sts_ch = Constraint(
        m.t,
        rule=c_sts_ch_rule,
        doc='q_ch [STS] = q_sco_hx1 + q_dch [LTS]')
        
    # Storage max charge constraint
    m.c_sto_max_ch = Constraint(
        m.t, m.sto,
        rule=c_sto_max_ch_rule,
        doc='q_ch <= max_charge_rate')
        
    # Storage max discharge constraint
    m.c_sto_max_dch = Constraint(
        m.t, m.sto,
        rule=c_sto_max_dch_rule,
        doc='q_dch <= max_discharge_rate')
        
    # Storage capacity constraint
    m.c_sto_cap = Constraint(
        m.t, m.sto,
        rule=c_sto_cap_rule,
        doc='q_sto <= capacity')
        
    # Storage stored energy constraint
    m.c_sto = Constraint(
        m.t, m.sto,
        rule=c_sto_rule,
        doc='q_sto(t+1) == q_sto(t) + q_ch(t) - q_dch(t)')
        
    # Percent charge calculation
    m.c_percent_charge = Constraint(
        m.t, m.sto,
        rule=c_percent_charge_rule,
        doc='percent_charge == q_sto(t)/max_cap')
        
    # STS - LTS constraint
    m.c_sto_ctrl_ch = Constraint(
        m.t,
        rule=c_sto_ctrl_ch_rule,
        doc='STS - LTS control - Charge situation')
        
    m.c_sto_ctrl_dch = Constraint(
        m.t,
        rule=c_sto_ctrl_dch_rule,
        doc='STS - LTS control - Discharge situation')
        
    m.c_delta_ch = Constraint(
        m.t,
        rule=c_delta_ch_rule,
        doc='delta_ch == y')
        
    m.c_delta_dch = Constraint(
        m.t,
        rule=c_delta_dch_rule,
        doc='delta_dch == (1-y)')
        
    # Dumping constrants
#    m.c_delta_sts = Constraint(
#        m.t,
#        rule=c_delta_sts_rule,
#        doc='%ch_sts >= delta_sts')
##        
#    m.c_delta_lts = Constraint(
#        m.t,
#        rule=c_delta_lts_rule,
#        doc='%ch_lts >= delta_lts')
#        
#    m.c_dump = Constraint(
#        m.t,
#        rule=c_dump_rule,
#        doc='q_dump == delta_extra * q_sco')
#        
#    m.c_delta_extra_1 = Constraint(
#        m.t,
#        rule=c_delta_extra_1_rule,
#        doc='-delta_sts + delta_extra <= 0')
#    
#    m.c_delta_extra_2 = Constraint(
#        m.t,
#        rule=c_delta_extra_2_rule,
#        doc='-delta_lts + delta_extra <= 0')
#        
#    m.c_delta_extra_3 = Constraint(
#        m.t,
#        rule=c_delta_extra_3_rule,
#        doc='delta_sts + delta_sts - delta_extra <= 1')
#        
#    m.c_sts_deltas = Constraint(
#        m.t,
#        rule=c_sts_deltas_rule,
#        doc='delta_sts_ch > delta_sts')
    
    # Demand constraint
    m.c_demand = Constraint(
        m.t,
        rule=c_demand_rule,
        doc='q_dch[STS] - q_ch[LTS] == demand + q_boi')
        
    # Objective
    # =========
    m.obj = Objective(
        rule=obj_rule,
        sense=minimize,
        doc='minimize boiler input')
        
    return m
    
"""
Constraints rules
"""

# Solar collector - hx1 constraint rule
def c_sco_hx1_rule(m, t):
    return (m.q_sco_hx1[t] == m.q_sco[t])
            
# Solar collector - storage charge constraint rule
def c_sts_ch_rule(m, t):
    return (m.q_ch[t, 'STS'] == (m.q_sco_hx1[t])/3600 + m.q_dch[t, 'LTS'])

# Storage max charge constraint rule
def c_sto_max_ch_rule(m, t, sto):
    if sto == 'LTS':
        return (m.q_ch[t, sto] <= m.delta_ch[t] *
                m.storage.loc[sto]['Max charge rate'])
    else:
        return (m.q_ch[t, sto] <= 
                m.storage.loc[sto]['Max charge rate'])
    
# Storage max discharge constraint rule
def c_sto_max_dch_rule(m, t, sto):
    if sto == 'LTS':
        return (m.q_dch[t, sto] <= m.delta_dch[t] * 
                m.storage.loc[sto]['Max discharge rate'])
    else:
        return (m.q_dch[t, sto] <= m.storage.loc[sto]['Max discharge rate'])
    
# Storage capacity constraint rule
def c_sto_cap_rule(m, t, sto):
    return (m.q_sto[t, sto] <= m.storage.loc[sto]['Capacity'])
    
# Percent charge calculation rule
def c_percent_charge_rule(m, t, sto):
    return (m.percent_charge[t, sto] == m.q_sto[t, sto] /
            m.storage.loc[sto]['Capacity'])
    
# Storage stored energy constraint rule
def c_sto_rule(m, t, sto):
    if t == 0:
        return (m.q_sto[t, sto] == m.storage.loc[sto]['Initial stored energy'])
    else:
        return (m.q_sto[t, sto] == m.q_sto[t-1, sto] + (m.q_ch[t-1, sto] -
                m.q_dch[t-1, sto] - m.storage.loc[sto]['Standing loss']) * 3600)

# STS - LTS interaction constraint rule
def c_sto_ctrl_ch_rule(m, t):
    return (m.percent_charge[t, 'STS'] - m.ch_req[t] <= 
            0.75 * m.y[t])
    
def c_sto_ctrl_dch_rule(m, t):
    return (m.percent_charge[t, 'STS'] - m.ch_req[t] >= 0.00001 + (-1) * 
            (1 - m.y[t]))
            
def c_delta_ch_rule(m,t):
    return (m.delta_ch[t] == m.y[t])
    
def c_delta_dch_rule(m,t):
    return (m.delta_dch[t] == (1 - m.y[t]))

# Dumping constraints rules
#def c_delta_sts_rule(m, t):
#    return (m.percent_charge[t, 'STS'] >= m.delta_sts[t])
#
#def c_delta_lts_rule(m, t):
#    return (m.percent_charge[t, 'LTS'] >= m.delta_lts[t])
    
#def c_dump_rule(m, t):
#    return (m.q_dump[t] == m.delta_extra[t] * m.q_sco[t])
#    
#def c_delta_extra_1_rule(m, t):
#    return (-m.delta_sts[t] + m.delta_extra[t] <= 0)
#    
#def c_delta_extra_2_rule(m, t):
#    return (-m.delta_lts[t] + m.delta_extra[t] <= 0)
#    
#def c_delta_extra_3_rule(m, t):
#    return (m.delta_sts[t] + m.delta_lts[t] - m.delta_extra[t] <= 1)
    
#def c_sts_deltas_rule(m, t):
#    return (m.delta_sts_ch[t] + m.delta_sts[t] == 1)

# Demand constraint rule
def c_demand_rule(m, t):
#    return (sum(m.q_dch[t, s] for s in m.storage.index) + m.q_boi[t] ==
#            m.demand.loc[t]['Demand'])
    return (m.q_dch[t, 'STS'] - m.q_ch[t, 'LTS'] + m.q_boi[t] ==
            m.demand.loc[t]['Demand'])

"""
Objective rule
"""
def obj_rule(m):
    return (sum(m.q_boi[t]for t in m.t))

"""
Main program
"""
if __name__ == '__main__':
    start = timeit.default_timer()
    
    input_data = read_excel('sdhData_3months.xlsx')
    
    model = create_model(input_data)

    model.pprint()

    instance = model.create()
    opt = SolverFactory("cplex")
    results = opt.solve(instance)

    instance.load(results)

    results.write()

    soc = []
    q_boi = []
    q_ch = []
    q_dch = []
    q_sco_hx1 = []
    
    for v in instance.active_components(Var):
        print("Variable", v)
        varobject = getattr(instance, v)
        
               
        for index in varobject:
            print(" ", index, varobject[index].value)
            if v == 'q_sto':
                soc.append(varobject[index].value)
            elif v == 'q_boi':
                q_boi.append(varobject[index].value)
            elif v == 'q_ch':
                q_ch.append(varobject[index].value)
            elif v == 'q_dch':
                q_dch.append(varobject[index].value)
            elif v == 'q_sco_hx1':
                q_sco_hx1.append(varobject[index].value)

    print "Objective value (NPV): ", results.Solution.Objective

    stop = timeit.default_timer()

    print "Solution time: ", stop-start, " seconds."
    
    soc_sts = soc[0::2]
    soc_lts = soc[1::2]
    
    q_ch_sts = q_ch[0::2]
    q_ch_lts = q_ch[1::2]
    
    q_dch_sts = q_dch[0::2]
    q_dch_lts = q_dch[1::2]
    
    """
    Plotting
    """
    demand = input_data['demand']['Demand'].tolist()
    ambient_temperature = input_data['weather']['Temperature'].tolist()
    
    fig1 = plt.figure()
    ax1 = fig1.gca()
    ind = np.arange(len(demand))
    bar11 = ax1.bar(ind, q_dch_sts, label='STS', color='g', align='center')
    bar12 = ax1.bar(ind, q_dch_lts, label='LTS', color='y', align='center', bottom=q_dch_sts)
    bar13 = ax1.bar(ind, q_boi, label='BOI', color='r', align='center', bottom= [sum(x) for x in zip(q_dch_sts, q_dch_lts)])
    
    plot11 = ax1.plot(ind, demand, label='Demand', marker='o')
#    plot12 = ax1.plot(ind, q_sco_hx1, label='Q sco hx1', marker='*')
    ax1.set_xlim(0, len(demand))
    ax1.set_ylim(0, max(demand)+500)
    ax1.set_ylabel('kW')
    
    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5, mode="expand", borderaxespad=0)
    
    fig2 = plt.figure()
    ax21 = subplot(311)
    ax21.plot(ind, q_ch_sts)
    ax21.set_ylabel('STS Charge (kW)')
    ax21.grid()
    
    ax22 = subplot(312, sharex=ax21)
    ax22.plot(ind, q_dch_sts)
    ax22.set_ylabel('STS Discharge (kW)')
    ax22.grid()
    
    ax23 = subplot(313, sharex=ax21)
    ax23.plot(ind, soc_sts)
    ax23.set_ylabel('STS State of Charge')
    ax23.set_xlabel('Hour')
    ax23.grid()
    
    fig3 = plt.figure()
    ax31 = subplot(311)
    ax31.plot(ind, q_ch_lts)
    ax31.set_ylabel('LTS Charge (kW)')
    ax31.grid()
    
    ax32 = subplot(312, sharex=ax31)
    ax32.plot(ind, q_dch_lts)
    ax32.set_ylabel('LTS Discharge (kW)')
    ax32.grid()
    
    ax33 = subplot(313, sharex=ax31)
    ax33.plot(ind, soc_lts)
    ax33.set_ylabel('LTS State of Charge')
    ax33.set_xlabel('Hour')
    ax33.grid()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    