# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:01:02 2015

@author: s1370831
"""

from __future__ import division
from pyomo.environ import *

import numpy as np
import pandas as pd

def read_excel(filename):
    """
    Read Excel input file which contains equipment data and connectivity
    matrix.
    """
    with pd.ExcelFile(filename) as xls:
        conversion = xls.parse(
            'Conversion',
            index_col=['Equipment'])
        demand = xls.parse(
            'Demand',
            index_col=['t'])
        connectivity = xls.parse(
            'ConnectivityMatrix')
        heat_matrix = xls.parse(
            'HeatMatrix')
        cool_matrix = xls.parse(
            'CoolMatrix')
        electricity_matrix = xls.parse(
            'ElectricityMatrix')
        economic = xls.parse(
            'Economic')
            
    data = {
        'conversion': conversion,
        'demand': demand,
        'connectivity': connectivity,
        'heat matrix': heat_matrix,
        'cool matrix': cool_matrix,
        'electricity matrix': electricity_matrix,
        'economic': economic}
        
    return data


def create_model(data):
    """
    Create a pyomo ConcreteModel from given input data
    """
    m = ConcreteModel(name="SDOMILP")
    
    m.timesteps = data['demand'].index.tolist()
    
    m.conversion = data['conversion']
    m.demand = data['demand']
    m.connectivity = data['connectivity']
    m.heat_matrix = data['heat matrix']
    m.cool_matrix = data['cool matrix']
    m.electricity_matrix = data['electricity matrix']
    m.economic = data['economic']
    
    m.heatrow = m.connectivity.loc['HD']
    m.heatsupplier = m.heatrow[m.heatrow == 1]
    
    m.coolrow = m.connectivity.loc['CD']
    m.coolsupplier = m.coolrow[m.coolrow == 1]
    
    m.elrow = m.connectivity.loc['EG']
    m.elsupplier = m.elrow[m.elrow == 1]
    
    # NOT A GOOD WAY TO IDENTIFY INTERNAL HEAT DEMAND (E.G. ABS CHILLER)    
    m.heatdemander = m.heat_matrix['BOI1'][m.heat_matrix['BOI1'] == 1]
    m.internal_heat_demand = m.heatdemander.drop('HD')
    
    # NOT A GOOD WAY TO IDENTIFY INTERNAL ELECTRICITY DEMAND
    m.eldemander = m.electricity_matrix['CHP1'][m.electricity_matrix['CHP1'] == 1]
    m.internal_el_demand = m.eldemander.drop('EG')
    
    
    # Parameters
    # ==========
    
    m.cft = Param(
        initialize=m.economic['cashflow-time'][0],
        within=NonNegativeReals,
        doc='Cashflow time')
    
    m.dr = Param(
        initialize=m.economic['discount-rate'][0],
        within=NonNegativeReals,
        doc='Discount rate')
    
    
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
        
    m.con_heat = Set(
        initialize=m.heatsupplier.index,
        ordered=True,
        doc='Set of conversion equipment supplying heat')
        
    m.int_heat_demand = Set(
        initialize=m.internal_heat_demand.index,
        ordered=True,
        doc='Set of equipment which require heat as an input')
        
    m.con_cool = Set(
        initialize=m.coolsupplier.index,
        ordered=True,
        doc='Set of conversion equipment supplying coolth')
        
    m.con_el = Set(
        initialize=m.elsupplier.index,
        ordered=True,
        doc='Set of conversion equipment supplying electricity')
        
    m.int_el_demand = Set(
        initialize=m.internal_el_demand.index,
        ordered=True,
        doc='Set of equipment require electricity as an input')
    
    # cost type
    m.cost_type = Set(
        initialize=['Inv', 'Mai', 'Fuel', 'Inc'])
    # Variables
    # =========
    
    m.size = Var(
        m.con,
        within=NonNegativeReals,
        doc='Conversion equipment size')
        
    m.y = Var(
        m.con,
        within=Binary,
        doc='Binary Install/Not')
    
    m.v = Var(
        m.t, m.con,
        within=NonNegativeReals,
        doc='Output power of the equipment')
        
    m.delta = Var(
        m.t, m.con,
        within=Binary,
        doc='ON/OFF status of the equipment')
    
    m.ksi = Var(
        m.t, m.con,
        within=NonNegativeReals,
        doc='ksi = delta * v')
    
    m.psi = Var(
        m.t, m.con,
        within=NonNegativeReals,
        doc='psi == equipment size for each t')
        
    m.el_sell = Var(
        m.t,
        within=NonNegativeReals,
        doc='Electricity sold to the grid')
        
    m.el_buy = Var(
        m.t,
        within=NonNegativeReals,
        doc='Electricity bought from the grid')
        
    # costs
    m.costs = Var(
        m.cost_type,
        within=NonNegativeReals,
        doc='Cost by type')
    
    # Expressions
    # ===========
               
    m.u = Var(
        m.t, m.con,
        within=NonNegativeReals,
        doc='Input power of the equipment')
    
    # Constraints
    # ===========
    # (constraints rules are defined separately)
    
    # Equipment size constraints
    m.c_size_min = Constraint(
        m.con,
        rule=c_size_min_rule,
        doc='min_cap <= size')
    
    m.c_size_max = Constraint(
        m.con,
        rule=c_size_max_rule,
        doc='size <= max_cap')
        
    m.c_peak_heat = Constraint(
        rule=c_peak_heat_rule,
        doc='size heat supplier == peak heat')
        
    m.c_peak_cool = Constraint(
        rule=c_peak_cool_rule,
        doc='size cool supplier == peak cool')
        
    # Output constraints
    m.c_v_min = Constraint(
        m.t, m.con,
        rule=c_v_min_rule,
        doc='ksi*v_min <= v')
        
    m.c_v_max = Constraint(
        m.t, m.con,
        rule=c_v_max_rule,
        doc='v <= ksi')
        
    # Delta (ON/OFF) constraints
    m.c_delta = Constraint(
        m.t, m.con,
        rule=c_delta_rule,
        doc='delta <= y')
        
    # ksi constraints
    m.c_ksi_min = Constraint(
        m.t, m.con,
        rule=c_ksi_min_rule,
        doc='delta*min-cap <= ksi')
        
    m.c_ksi_max = Constraint(
        m.t, m.con,
        rule=c_ksi_max_rule,
        doc='ksi <= delta*max-cap')
        
    # psi constraints
    m.c_psi = Constraint(
        m.t, m.con,
        rule=c_psi_rule,
        doc='psi == size')
        
    m.c_psi_min = Constraint(
        m.t, m.con,
        rule=c_psi_min_rule,
        doc='0 <= psi-ksi')
        
    m.c_psi_max = Constraint(
        m.t, m.con,
        rule=c_psi_max_rule,
        doc='psi-ksi <= (1-delta)*max-cap')
        
#    # Combinatorial redundancy constraint
#    m.c_combinatorial_redundancy = Constraint(
#        m.con,
#        rule=c_combinatorial_redundancy_rule,
#        doc='y(n+1) <= y(n)')
#        
    # Demand constraints
    m.c_heatdemand = Constraint(
        m.t,
        rule=c_heatdemand_rule,
        doc='heat supply == heat demand')
    
    m.c_cooldemand = Constraint(
        m.t,
        rule=c_cooldemand_rule,
        doc='cool supply == cool demand')
        
    m.c_eldemand = Constraint(
        m.t,
        rule=c_eldemand_rule,
        doc='electricity demand == electricity supply')
    
    # Input constraints
    m.c_input = Constraint(
        m.t, m.con,
        rule=c_input_rule,
        doc='equipment input calculation function')
        
    # Costs constraint
    m.c_costs = Constraint(
        m.cost_type,
        rule=c_costs_rule,
        doc='cost function by cost type')
    
    # Objective
    m.obj = Objective(
        rule=obj_rule,
        sense=maximize,
        doc='maximize NPV')
    
    return m

"""
Constraints rule
"""

# Size constraints
def c_size_min_rule(m, con):
    return (m.y[con] * m.conversion.loc[con]['min-cap'] <= m.size[con])

def c_size_max_rule(m, con):
    return (m.size[con] <= m.y[con] * m.conversion.loc[con]['max-cap'])
    
def c_peak_heat_rule(m):
    return (sum(m.size[p] for p in m.con_heat) == m.demand['peak-heat'][0])
    
def c_peak_cool_rule(m):
    return (sum(m.size[q] for q in m.con_cool) == m.demand['peak-cool'][0])

# Output constraints
def c_v_min_rule(m, t, con):
    return (m.ksi[t, con] * m.conversion.loc[con]['min-lf'] <= m.v[t, con])

def c_v_max_rule(m, t, con):
    return (m.v[t, con] <= m.ksi[t, con])

# Delta (ON/OFF) constraint
def c_delta_rule(m, t, con):
    return (m.delta[t, con] <= m.y[con])
    
# Ksi constraints
def c_ksi_min_rule(m, t, con):
    return (m.delta[t, con]*m.conversion.loc[con]['min-cap'] <= m.ksi[t, con])
    
def c_ksi_max_rule(m, t, con):
    return (m.ksi[t, con] <= m.delta[t, con]*m.conversion.loc[con]['max-cap'])

# Psi constraints
def c_psi_rule(m, t, con):
    return (m.psi[t, con] == m.size[con])
    
def c_psi_min_rule(m, t, con):
    return (0 <= (m.psi[t, con] - m.ksi[t, con]))

def c_psi_max_rule(m, t, con):
    return ((m.psi[t, con] - m.ksi[t, con]) <= (1-m.delta[t, con]) *
            m.conversion.loc[con]['max-cap'])

# Demand constraints
def c_heatdemand_rule(m, t):  
    return (sum(m.v[t, c] for c in m.con_heat)  == m.demand.loc[t]['Heating']
            + (sum(m.u[t, d] for d in m.int_heat_demand)))
    
def c_cooldemand_rule(m, t):
    return (sum(m.v[t, c] for c in m.con_cool)  == m.demand.loc[t]['Cooling'])

def c_eldemand_rule(m, t):
    return (sum(m.v[t, c] * m.conversion.loc[c]['el-efficiency'] 
        for c in m.con_el) + m.el_buy[t] == 
        sum(m.u[t, d] for d in m.int_el_demand) + m.el_sell[t])
#    return Constraint.Skip
    
# Input constraints
def c_input_rule(m, t, con):
    return (m.u[t, con] == (m.ksi[t, con] * m.conversion.loc[con]['u0'] / 
            m.conversion.loc[con]['efficiency']) + (((m.v[t, con] * 
            m.conversion.loc[con]['el-efficiency']) /
            m.conversion.loc[con]['efficiency']) *
            m.conversion.loc[con]['performance-slope']))
            
# Costs constraints (calculations)
def c_costs_rule(m, cost_type):
    """
    Calculate total costs by cost type
    """
    if cost_type == 'Inv':
        return m.costs['Inv'] == \
            sum(m.y[p] * m.conversion.loc[p]['I0'] + m.size[p] * 
                m.conversion.loc[p]['investment-slope-1'] for p in m.con)
        
    elif cost_type == 'Mai':
        return m.costs['Mai'] == \
            sum((m.conversion.loc[p]['maint-percentage']/100) * 
                (m.y[p] * m.conversion.loc[p]['I0'] + m.size[p] * 
                m.conversion.loc[p]['investment-slope-1']) for p in m.con)
        
    elif cost_type == 'Fuel':
        gas_user_index = m.connectivity['NG'][m.connectivity['NG'] == 1].index
#        el_user_index = m.connectivity['EG'][m.connectivity['EG'] == 1].index
        return m.costs['Fuel'] == \
            ((sum(m.u[t, p] for t in m.t for p in gas_user_index) * 
                m.economic['gas-tariff'][0]) + \
            (sum(m.el_buy[t] * m.economic['el-tariff'][0] for t in m.t)) - \
            (sum(m.el_sell[t] * m.economic['feedin-tariff'][0] for t in m.t))) * \
            91 * 24
        
    elif cost_type == 'Inc':
        return Constraint.Skip
        
    else:
        raise NotImplementedError("Unknown cost type.")

# Objective rule
def obj_rule(m):
    return (((((m.dr + 1)**m.cft) - 1) / (m.dr * ((m.dr + 1)**m.cft))) * 
        -summation(m.costs))
    

"""
Main program
"""
if __name__ == '__main__':
    
    input_data = read_excel('superstructureIO.xlsx')
    
    model = create_model(input_data)
    
    model.pprint()
    
    instance = model.create()
    opt = SolverFactory("glpk")
    results = opt.solve(instance)
    
    instance.load(results)
    
    for v in instance.active_components(Var):
        print("Variable", v)
        varobject = getattr(instance, v)
        for index in varobject:
            print(" ", index, varobject[index].value)
    
    print "Objective value (NPV): ", results.Solution.Objective
