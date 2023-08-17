'''
FIG2_4.PY - Compare the punisher's performance with that of gene overlaps.
'''
# By Kirill Sechkar

# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------
# multiprocessing - must be imported and handled first!
import os
import multiprocessing
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(multiprocessing.cpu_count())

import numpy as np
import jax
import jax.numpy as jnp
import functools
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController, SteadyStateEvent
import pandas as pd
from bokeh import plotting as bkplot, models as bkmodels, layouts as bklayouts
from bokeh.colors import RGB as bkRGB
from math import pi
import pickle

import time

# CIRCUIT AND EXTERNAL INPUT IMPORTS -----------------------------------------------------------------------------------
import synthetic_circuits_jax as circuits
from cell_model import *


# GENERATE FIGURE FOR THE LOSS OF ONE TOGGLE SWITCH --------------------------------------------------------------------
def get_growth_rate(par,  # dictionary with model parameters
                    rel_kplus_cat,
                    default_sgp4j,  # default synthetic gene parameters in jax.array form
                    ode_with_circuit,  # ODE function for the cell with synthetic circuit
                    circuit_F_calc,  # function for calculating the circuit's reaction rates
                    init_conds,  # initial condition DICTIONARY
                    circuit_genes, circuit_miscs, circuit_name2pos,
                    # dictionaries with circuit gene and miscellaneous specie names, species name to vector position decoder
                    tf=(0,48), rtol=1e-6, atol=1e-6 , # simulation parameters: maximum time frame, relative and absolute tolerances
                    ss_rtol=0.0001, ss_atol=0.0001 # steady state determination tolerances
                    ):
    sgp4j= (default_sgp4j[0].at[-4:].set(default_sgp4j[0][-1]*rel_kplus_cat), default_sgp4j[1], default_sgp4j[2], default_sgp4j[3])

    cellmodel_auxil= CellModelAuxiliary()
    # Simulate the cell model t find the steady state
    vector_field = lambda t, y, args: ode_with_circuit(t, y, args)
    term = ODETerm(vector_field)
    args = (
        par,  # model parameters
        circuit_name2pos,  # gene name - position in circuit vector decoder
        len(circuit_genes), len(circuit_miscs),  # number of genes and miscellaneous species in the circuit
        sgp4j  # relevant synthetic gene parameters in jax.array form
    )
    sol = diffeqsolve(term,
                      solver=Dopri5(),
                      args=args,
                      t0=tf[0], t1=tf[1], dt0=0.1,
                      y0=cellmodel_auxil.x0_from_init_conds(init_conds, circuit_genes, circuit_miscs),
                      max_steps=None,
                      stepsize_controller=PIDController(rtol=rtol, atol=atol),
                      discrete_terminating_event=SteadyStateEvent(ss_rtol, ss_atol))
    x=sol.ys[-1]
    # give the state vector entries meaningful names
    m_a = x[0]  # metabolic gene mRNA
    m_r = x[1]  # ribosomal gene mRNA
    p_a = x[2]  # metabolic proteins
    R = x[3]  # non-inactivated ribosomes
    tc = x[4]  # charged tRNAs
    tu = x[5]  # uncharged tRNAs
    s = x[6]  # nutrient quality (constant)
    h = x[7]  # chloramphenicol concentration (constant)
    x_het = x[8:8 + 2 * len(circuit_genes)]  # heterologous protein concentrations
    misc = x[8 + 2 * len(circuit_genes):8 + 2 * len(circuit_genes) + len(circuit_miscs)]  # miscellaneous species

    # vector of Synthetic Gene Parameters 4 JAX
    kplus_het, kminus_het, n_het, d_het = sgp4j

    # CALCULATE PHYSIOLOGICAL VARIABLES
    # translation elongation rate
    e = e_calc(par, tc)

    # ribosome dissociation constants
    k_a = k_calc(e, par['k+_a'], par['k-_a'], par['n_a'])  # metabolic genes
    k_r = k_calc(e, par['k+_r'], par['k-_r'], par['n_r'])  # ribosomal genes
    k_het = k_calc(e,kplus_het,kminus_het,n_het)  # heterologous genes

    T = tc / tu  # ratio of charged to uncharged tRNAs

    prodeflux = jnp.sum(d_het * n_het * x_het[len(circuit_genes):2 * len(circuit_genes)])

    # resource competition denominator
    Q = Q_calc(par, e, R, h, prodeflux)  # scaling factor to omit modelling housekeeping genes
    D = ((par['K_D'] + h) / par['K_D']) * \
        (1 + Q * (m_a / k_a + m_r / k_r + (x_het[0:len(circuit_genes)] / k_het).sum()))
    B = R * (par['K_D'] / (
                par['K_D'] + h) - 1 / D)  # actively translating ribosomes (inc. those translating housekeeping genes)

    l = l_calc(par, e, B, prodeflux)  # growth rate
    return l


# MAIN FUNCTION (FOR CALLING THE FIGURE GENERATION FUNCTIONS) ----------------------------------------------------------
def main():
    # set up jax
    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update("jax_enable_x64", True)
    print(jax.lib.xla_bridge.get_backend().platform)
    print(len(jax.devices()))

    # TOGGLES+PUNISHER: CIRCUIT SPECIFICATION --------------------------------------------------------------------------
    # initialise cell model
    cellmodel_auxil = CellModelAuxiliary()  # auxiliary tools for simulating the model and plotting simulation outcomes
    par = cellmodel_auxil.default_params()  # get default parameter values
    init_conds = cellmodel_auxil.default_init_conds(par)  # get default initial conditions

    # load synthetic gene circuit
    ode_with_circuit, circuit_F_calc, par, init_conds, circuit_genes, circuit_miscs, circuit_name2pos, circuit_styles = cellmodel_auxil.add_circuit(
        circuits.twotoggles_punisher_initialise,
        circuits.twotoggles_punisher_ode,
        circuits.twotoggles_punisher_F_calc,
        par, init_conds)  # load the circuit

    # toggle switch parameters
    for togswitchnum in (1, 2):  # cycle through toggle switches
        for toggenenum in (1, 2):  # cycle through the genes of the current switch
            par['c_tog' + str(togswitchnum) + str(toggenenum)] = 1  # copy no. (nM)
            par['a_tog' + str(togswitchnum) + str(toggenenum)] = 10 ** 5 / 2  # promoter strength (unitless)

            # transcription regulation function
            reg_func_string = 'dna(tog' + str(togswitchnum) + str(toggenenum) + '):p_tog' + str(togswitchnum) + str(
                (toggenenum - 2) % 2 + 1)  # dna(rep1):p_rep3, dna(rep2):p_rep1 or dna(rep3):p_rep2
            par['K_' + reg_func_string] = 2000  # half-saturation constant
            par['eta_' + reg_func_string] = 2  # Hill coefficient
            par['baseline_tog' + str(togswitchnum) + str(
                toggenenum)] = 0.01  # baseline transcription activation due to leakiness
            par['p_tog' + str(togswitchnum) + str(
                toggenenum) + '_ac_frac'] = 1  # active fraction of protein (i.e. share of molecules not bound by the inducer)

        # break symmetry for each of the toggle switches
        init_conds['m_tog' + str(togswitchnum) + '1'] = 4000

    # Punisher parameters
    par['c_switch'] = 10.0  # gene concentration (nM)
    par['a_switch'] = 100.0  # promoter strength (unitless)
    par['c_int'] = 10.0  # gene concentration (nM)
    par['a_int'] = 60.0  # promoter strength (unitless)
    par['d_int'] = 6.0  # integrase protein degradation rate (to avoid unnecessary punishment)
    par['c_cat'] = 10.0  # gene concentration (nM)
    par['a_cat'] = 500.0  # promoter strength (unitless)

    # punisher's transcription regulation function
    par['K_switch'] = 625.0  # Half-saturation constant for the self-activating switch gene promoter (nM)
    par['eta_switch'] = 2

    # Hill coefficient for the self-activating switch gene promoter (unitless)
    par['baseline_switch'] = 0.025  # Baseline value of the switch gene's transcription activation function
    par['p_switch_ac_frac'] = 0.83  # active fraction of protein (i.e. share of molecules bound by the inducer)

    # culture medium
    init_conds['s'] = 0.3
    par['h_ext'] = 10.5 * (10.0 ** 3)

    # TOGGLES+PUNISHER: SIMULATION ------------------------------------------------------------------------------------
    simulate=False
    if(simulate):
        default_sgp4j = cellmodel_auxil.synth_gene_params_for_jax(par, circuit_genes)
        get_growth_rate_for_param = lambda par: get_growth_rate(par,  # dictionary with model parameters
                        1,
                        default_sgp4j,  # default synthetic gene parameters in jax.array form
                        ode_with_circuit,  # ODE function for the cell with synthetic circuit
                        circuit_F_calc,  # function for calculating the circuit's reaction rates
                        init_conds,  # initial condition DICTIONARY
                        circuit_genes, circuit_miscs, circuit_name2pos,
                        # dictionaries with circuit gene and miscellaneous specie names, species name to vector position decoder
                        tf=(0,96), rtol=1e-6, atol=1e-6 , # simulation parameters: maximum time frame, relative and absolute tolerances
                        ss_rtol=0.0001, ss_atol=0.0001 # steady state determination tolerances
                        )

        # get growth rate for both toggles present
        print('Both toggles...')
        pun_growth_rate_bothtoggles = float(get_growth_rate_for_param(par))
        # get growth rate with a toggle and a half
        print('Toggle and a half...')
        par['func_tog11']=0
        pun_growth_rate_toggleandhalf = float(get_growth_rate_for_param(par))
        # get growth rate with one toggle
        print('One toggle...')
        par['func_tog11']=0
        par['func_tog12']=0
        pun_growth_rate_onetoggle = float(get_growth_rate_for_param(par))
        # get growth rate with two halves of a toggles
        print('Two halves of a toggle...')
        par['func_tog11']=0
        par['func_tog12']=1
        par['func_tog21']=0
        pun_growth_rate_twohalftoggles = float(get_growth_rate_for_param(par))
        # get growth rate with half a toggle
        print('Half a toggle...')
        par['func_tog11']=0
        par['func_tog12']=0
        par['func_tog21']=0
        pun_growth_rate_halftoggle = float(get_growth_rate_for_param(par))
        # get growth rate with no toggles
        print('No toggles...')
        par['func_tog11']=0
        par['func_tog12']=0
        par['func_tog21']=0
        par['func_tog22']=0
        pun_growth_rate_notoggles = float(get_growth_rate_for_param(par))

    # ADDICTIVE TOGGLES: CIRCUIT SPECIFICATION -------------------------------------------------------------------------
    # initialise cell model
    add_par = cellmodel_auxil.default_params()  # get default parameter values
    add_init_conds = cellmodel_auxil.default_init_conds(add_par)  # get default initial conditions

    # load synthetic gene circuit
    add_ode_with_circuit, add_circuit_F_calc, add_par, add_init_conds, \
        add_circuit_genes, add_circuit_miscs, add_circuit_name2pos, add_circuit_styles = cellmodel_auxil.add_circuit(
        circuits.two_addictive_toggles_initialise,
        circuits.two_addictive_toggles_ode,
        circuits.two_addictive_toggles_F_calc,
        add_par, add_init_conds)  # load the circuit

    # toggle switch parameters - made identical to the case with the punisher
    for togswitchnum in (1, 2):  # cycle through toggle switches
        for toggenenum in (1, 2):  # cycle through the genes of the current switch
            add_par['c_tog' + str(togswitchnum) + str(toggenenum)] = par['c_tog' + str(togswitchnum) + str(toggenenum)]
            add_par['a_tog' + str(togswitchnum) + str(toggenenum)] = par['a_tog' + str(togswitchnum) + str(toggenenum)]
            add_par['b_tog' + str(togswitchnum) + str(toggenenum)] = par['b_tog' + str(togswitchnum) + str(toggenenum)]
            add_par['k+_tog' + str(togswitchnum) + str(toggenenum)] = par['k+_tog' + str(togswitchnum) + str(toggenenum)]
            add_par['k-_tog' + str(togswitchnum) + str(toggenenum)] = par['k-_tog' + str(togswitchnum) + str(toggenenum)]
            add_par['n_tog' + str(togswitchnum) + str(toggenenum)] = par['n_tog' + str(togswitchnum) + str(toggenenum)]
            add_par['d_tog' + str(togswitchnum) + str(toggenenum)] = par['d_tog' + str(togswitchnum) + str(toggenenum)]

            # transcription regulation function
            reg_func_string = 'dna(tog' + str(togswitchnum) + str(toggenenum) + '):p_tog' + str(togswitchnum) + str(
                (toggenenum - 2) % 2 + 1)  # dna(rep1):p_rep3, dna(rep2):p_rep1 or dna(rep3):p_rep2
            add_par['K_' + reg_func_string] = par['K_' + reg_func_string]
            add_par['eta_' + reg_func_string] = par['eta_' + reg_func_string]
            add_par['baseline_tog' + str(togswitchnum) + str(toggenenum)] = par['baseline_tog' + str(togswitchnum) + str(toggenenum)]
            add_par['p_tog' + str(togswitchnum) + str(toggenenum) + '_ac_frac'] = par['p_tog' + str(togswitchnum) + str(toggenenum) + '_ac_frac']

        # initial conditions also identical to the case with the punisher
        init_conds['m_tog' + str(togswitchnum) + '1'] = 500

    # overlapping cat genes' parameters
    # NOTE: a_catXY and c_catXY will be DISREGARDED due to co-transcription. a_togXY and c_togXY are used instead !
    for togswitchnum in (1, 2):  # cycle through toggle switches
        for toggenenum in (1, 2):  # cycle through the genes of the current switch
            add_par['k+_cat' + str(togswitchnum) + str(toggenenum)] = add_par['k+_tog' + str(togswitchnum) + str(toggenenum)] * 0.025

    # culture medium
    add_init_conds['s'] = init_conds['s']
    add_par['h_ext'] = par['h_ext']

    # ADDICTIVE TOGGLES: DEFINE TESTED RANGE OF RBS STRENGTHS ----------------------------------------------------------
    default_sgp4j=cellmodel_auxil.synth_gene_params_for_jax(add_par,add_circuit_genes)
    # define the tested range of RBS strengths
    rel_kplus_cats=jnp.logspace(-1.25,-0.5,19)
    # redefine the parameter dictionary for pmapping
    addpar_for_pmap=add_par.copy()
    for togswitchnum in (1, 2):  # cycle through toggle switches
        for toggenenum in (1, 2):  # cycle through the genes of the current switch
            addpar_for_pmap['k+_cat' + str(togswitchnum) + str(toggenenum)] = rel_kplus_cats*add_par['k+_cat11']
    # make a pmapping axes dictionary
    pmap_for_addpar={}
    for parkey in add_par.keys():
        if(parkey[0:6]=='k+_cat'):
            pmap_for_addpar[parkey]=0
        else:
            pmap_for_addpar[parkey]=None

    if(simulate):
        get_growth_rate_for_params = lambda params, rel_kplus_cat: get_growth_rate(params,  # dictionary with model parameters
                                                                    rel_kplus_cat, # relative RBS strength for cat genes
                                                                    default_sgp4j,  # dictionary with synthetic gene parameters
                        add_ode_with_circuit,  # ODE function for the cell with synthetic circuit
                        add_circuit_F_calc,  # function for calculating the circuit's reaction rates
                        add_init_conds,  # initial condition DICTIONARY
                        add_circuit_genes, add_circuit_miscs, add_circuit_name2pos,
                        # dictionaries with circuit gene and miscellaneous specie names, species name to vector position decoder
                        tf=(0,96), rtol=1e-6, atol=1e-6 , # simulation parameters: maximum time frame, relative and absolute tolerances
                        ss_rtol=0.001, ss_atol=0.001 # steady state determination tolerances
                        )

        pmapped_growth_rates=jax.pmap(get_growth_rate_for_params,in_axes=(pmap_for_addpar,0))

        # ADDICTIVE TOGGLES: RUN SIMULATIONS -------------------------------------------------------------------------------
        # get growth rates with both toggles present
        print('Both toggles...')
        growth_rates_bothtoggles=np.array(pmapped_growth_rates(addpar_for_pmap,rel_kplus_cats))
        # get growth rates with a toggle and a half
        print('Toggle and a half...')
        addpar_for_pmap['func_tog11']=0
        growth_rates_toggleandhalf=np.array(pmapped_growth_rates(addpar_for_pmap,rel_kplus_cats))
        # get growth rates with one toggle present
        print('One toggle...')
        addpar_for_pmap['func_tog11']=0
        addpar_for_pmap['func_tog12']=0
        growth_rates_onetoggle=np.array(pmapped_growth_rates(addpar_for_pmap,rel_kplus_cats))
        # get growth rate with two halves of a toggles
        print('Two halves of a toggle...')
        addpar_for_pmap['func_tog11'] = 0
        addpar_for_pmap['func_tog12'] = 1
        addpar_for_pmap['func_tog21'] = 0
        growth_rates_twohalftoggles=np.array(pmapped_growth_rates(addpar_for_pmap,rel_kplus_cats))
        # get growth rates with half a toggle present
        print('Half a toggle...')
        addpar_for_pmap['func_tog11']=0
        addpar_for_pmap['func_tog12']=0
        addpar_for_pmap['func_tog21']=0
        growth_rates_halftoggle=np.array(pmapped_growth_rates(addpar_for_pmap,rel_kplus_cats))
        # get growth rates with no toggles present
        print('No toggles...')
        addpar_for_pmap['func_tog11']=0
        addpar_for_pmap['func_tog12']=0
        addpar_for_pmap['func_tog21']=0
        addpar_for_pmap['func_tog22']=0
        growth_rates_notoggles=np.array(pmapped_growth_rates(addpar_for_pmap,rel_kplus_cats))

        # pickle the results
        pickle.dump([rel_kplus_cats,
                     pun_growth_rate_bothtoggles,pun_growth_rate_toggleandhalf,pun_growth_rate_onetoggle,pun_growth_rate_twohalftoggles,pun_growth_rate_halftoggle,pun_growth_rate_notoggles,
                     growth_rates_bothtoggles,growth_rates_toggleandhalf,growth_rates_onetoggle,growth_rates_twohalftoggles,growth_rates_halftoggle,growth_rates_notoggles],open('fig4.pickle','wb'))
    else:
        # load pickled results
        rel_kplus_cats, \
            pun_growth_rate_bothtoggles, pun_growth_rate_toggleandhalf, pun_growth_rate_onetoggle, pun_growth_rate_twohalftoggles, pun_growth_rate_halftoggle, pun_growth_rate_notoggles,   \
            growth_rates_bothtoggles,growth_rates_toggleandhalf,growth_rates_onetoggle,growth_rates_twohalftoggles, growth_rates_halftoggle,growth_rates_notoggles=pickle.load(open('fig4.pickle','rb'))

    # PLOT -------------------------------------------------------------------------------------------------------------
    print('Plotting...')
    # set svg backed
    bkplot.output_file(filename="fig4.html",
                       title="Growth rates with addiction")  # set up bokeh output file
    font_size='8pt' # font size for the plots
    np_kplus_cats = np.array(rel_kplus_cats * add_par['k+_cat11'])

    # find region where overlaps fail
    fail_indices = np.where(
        np.logical_or(
            np.divide(growth_rates_twohalftoggles, growth_rates_bothtoggles) > 1,
            np.logical_or(
                np.logical_or(np.divide(growth_rates_toggleandhalf, growth_rates_bothtoggles) > 1,
                              np.divide(growth_rates_onetoggle, growth_rates_bothtoggles) > 1),
                np.logical_or(np.divide(growth_rates_halftoggle, growth_rates_bothtoggles) > 1,
                              np.divide(growth_rates_notoggles, growth_rates_bothtoggles) > 1)
        )))[0]
    # failure regions are on either side of the functional region as growth rates change monotonically
    if(len(fail_indices)>1):
        i=0
        while(fail_indices[i+1]-fail_indices[i]==1 and i<len(fail_indices)-1):
            i+=1
        fail_indices_left=fail_indices[0:i+1]
        fail_indices_right=fail_indices[i+1:]
    else:
        fail_indices_left=np.array([])
        fail_indices_right=np.array([])
    fail_kplus_cats_left=np_kplus_cats[fail_indices_left]
    fail_kplus_cats_right=np_kplus_cats[fail_indices_right]

    #
    # toggle and a half
    toggleandhalf_fig= bkplot.figure(
        frame_width=160,
        frame_height=160,
        title='tog11 mutated',
        x_axis_label="Overlapped cat RBS:ribosome \n dissoc. rate, 1/(nM*h)",
        y_axis_label="Rel. growth rate after mutation",
        x_range=(np_kplus_cats[0], np_kplus_cats[-1]),
        y_range=(0.95,1.01),
        y_axis_type="log",
        tools="box_zoom,pan,hover,reset,save"
    )
    toggleandhalf_fig.output_backend = "svg"
    # Addictive toggles
    toggleandhalf_fig.line(np_kplus_cats,np.divide(growth_rates_toggleandhalf,growth_rates_bothtoggles),
                    line_color=bkRGB(222, 49, 99),line_width=2,legend_label='Overlaps')
    # Punisher
    toggleandhalf_fig.line(np_kplus_cats,
                    pun_growth_rate_toggleandhalf / pun_growth_rate_bothtoggles * np.ones(len(np_kplus_cats)),
                    line_color=bkRGB(255, 103, 0), line_width=2, legend_label='Punisher')
    # Relative growth one line
    toggleandhalf_fig.line(np_kplus_cats, np.ones(len(np_kplus_cats)),
                    line_color=bkRGB(0, 0, 0), line_width=2, line_dash='dashed')
    # shade region where overlaps fail - left
    toggleandhalf_fig.add_layout(
        bkmodels.PolyAnnotation(xs=np.concatenate((fail_kplus_cats_left,np.flip(fail_kplus_cats_left))),
                                ys=np.concatenate((0.1*np.ones(len(fail_kplus_cats_left)),2*np.ones(len(fail_kplus_cats_left)))),
                                line_width=0, line_alpha=0,
                                fill_color=bkRGB(100, 100, 100, 0.25)))
    # shade region where overlaps fail - right
    toggleandhalf_fig.add_layout(
        bkmodels.PolyAnnotation(xs=np.concatenate((fail_kplus_cats_right,np.flip(fail_kplus_cats_right))),
                                ys=np.concatenate((0.1*np.ones(len(fail_kplus_cats_right)),2*np.ones(len(fail_kplus_cats_right)))),
                                line_width=0, line_alpha=0,
                                fill_color=bkRGB(100, 100, 100, 0.25)))

    # font and axis settings
    toggleandhalf_fig.xaxis.ticker = bkmodels.BasicTicker(desired_num_ticks=5)
    toggleandhalf_fig.xaxis.axis_label_text_font_size = font_size
    toggleandhalf_fig.xaxis.major_label_text_font_size = font_size
    toggleandhalf_fig.yaxis.axis_label_text_font_size = font_size
    toggleandhalf_fig.yaxis.major_label_text_font_size = font_size
    toggleandhalf_fig.legend.label_text_font_size = font_size
    toggleandhalf_fig.title.text_font_size = font_size
    #toggleandhalf_fig.legend.visible=False
    toggleandhalf_fig.legend.background_fill_alpha = 1
    toggleandhalf_fig.legend.location = "bottom_right"

    #
    # one toggle
    onetoggle_fig= bkplot.figure(
        frame_width=160,
        frame_height=160,
        title='tog11, tog12 mutated',
        x_axis_label="Overlapped cat RBS:ribosome \n dissoc. rate, 1/(nM*h)",
        y_axis_label="Rel. growth rate after mutation",
        x_range=(np_kplus_cats[0], np_kplus_cats[-1]),
        y_axis_type="log",
        tools="box_zoom,pan,hover,reset,save"
    )
    onetoggle_fig.output_backend = "svg"
    # Addictive toggles
    onetoggle_fig.line(np_kplus_cats,np.divide(growth_rates_onetoggle,growth_rates_bothtoggles),
                    line_color=bkRGB(222, 49, 99),line_width=2,legend_label='Overlaps')
    # Punisher
    onetoggle_fig.line(np_kplus_cats,
                    pun_growth_rate_onetoggle / pun_growth_rate_bothtoggles * np.ones(len(np_kplus_cats)),
                    line_color=bkRGB(255, 103, 0), line_width=2, legend_label='Punisher')
    # Relative growth one line
    onetoggle_fig.line(np_kplus_cats, np.ones(len(np_kplus_cats)),
                    line_color=bkRGB(0, 0, 0), line_width=2, line_dash='dashed')
    # shade region where overlaps fail - left
    onetoggle_fig.add_layout(
        bkmodels.PolyAnnotation(xs=np.concatenate((fail_kplus_cats_left, np.flip(fail_kplus_cats_left))),
                                ys=np.concatenate(
                                    (0.1 * np.ones(len(fail_kplus_cats_left)), 2 * np.ones(len(fail_kplus_cats_left)))),
                                line_width=0, line_alpha=0,
                                fill_color=bkRGB(100, 100, 100, 0.25)))
    # shade region where overlaps fail - right
    onetoggle_fig.add_layout(
        bkmodels.PolyAnnotation(xs=np.concatenate((fail_kplus_cats_right, np.flip(fail_kplus_cats_right))),
                                ys=np.concatenate((0.1 * np.ones(len(fail_kplus_cats_right)),
                                                   2 * np.ones(len(fail_kplus_cats_right)))),
                                line_width=0, line_alpha=0,
                                fill_color=bkRGB(100, 100, 100, 0.25)))

    # font and axis settings
    onetoggle_fig.xaxis.ticker = bkmodels.BasicTicker(desired_num_ticks=5)
    onetoggle_fig.xaxis.axis_label_text_font_size = font_size
    onetoggle_fig.xaxis.major_label_text_font_size = font_size
    onetoggle_fig.yaxis.axis_label_text_font_size = font_size
    onetoggle_fig.yaxis.major_label_text_font_size = font_size
    onetoggle_fig.legend.label_text_font_size = font_size
    onetoggle_fig.title.text_font_size = font_size
    #onetoggle_fig.legend.visible=False
    onetoggle_fig.legend.background_fill_alpha=1
    onetoggle_fig.legend.location= "right"
    #
    # two halves of a toggle
    twohalftoggles_fig= bkplot.figure(
        frame_width=160,
        frame_height=160,
        title='tog11, tog21 mutated',
        x_axis_label="Overlapped cat RBS:ribosome \n dissoc. rate, 1/(nM*h)",
        y_axis_label="Rel. growth rate after mutation",
        x_range=(np_kplus_cats[0], np_kplus_cats[-1]),
        y_axis_type="log",
        tools="box_zoom,pan,hover,reset,save"
    )
    twohalftoggles_fig.output_backend = "svg"
    # Addictive toggles
    twohalftoggles_fig.line(np_kplus_cats,np.divide(growth_rates_twohalftoggles,growth_rates_bothtoggles),
                    line_color=bkRGB(222, 49, 99),line_width=2,legend_label='Overlaps')
    # Punisher
    twohalftoggles_fig.line(np_kplus_cats,
                    pun_growth_rate_twohalftoggles / pun_growth_rate_bothtoggles * np.ones(len(np_kplus_cats)),
                    line_color=bkRGB(255, 103, 0), line_width=2, legend_label='Punisher')
    # Relative growth one line
    twohalftoggles_fig.line(np_kplus_cats, np.ones(len(np_kplus_cats)),
                    line_color=bkRGB(0, 0, 0), line_width=2, line_dash='dashed')
    # shade region where overlaps fail - left
    twohalftoggles_fig.add_layout(
        bkmodels.PolyAnnotation(xs=np.concatenate((fail_kplus_cats_left, np.flip(fail_kplus_cats_left))),
                                ys=np.concatenate(
                                    (0.1 * np.ones(len(fail_kplus_cats_left)), 2 * np.ones(len(fail_kplus_cats_left)))),
                                line_width=0, line_alpha=0,
                                fill_color=bkRGB(100, 100, 100, 0.25)))
    # shade region where overlaps fail - right
    twohalftoggles_fig.add_layout(
        bkmodels.PolyAnnotation(xs=np.concatenate((fail_kplus_cats_right, np.flip(fail_kplus_cats_right))),
                                ys=np.concatenate((0.1 * np.ones(len(fail_kplus_cats_right)),
                                                    2 * np.ones(len(fail_kplus_cats_right)))),
                                line_width=0, line_alpha=0,
                                fill_color=bkRGB(100, 100, 100, 0.25)))

    # font and axis settings
    twohalftoggles_fig.xaxis.ticker = bkmodels.BasicTicker(desired_num_ticks=5)
    twohalftoggles_fig.xaxis.axis_label_text_font_size = font_size
    twohalftoggles_fig.xaxis.major_label_text_font_size = font_size
    twohalftoggles_fig.yaxis.axis_label_text_font_size = font_size
    twohalftoggles_fig.yaxis.major_label_text_font_size = font_size
    twohalftoggles_fig.legend.label_text_font_size = font_size
    twohalftoggles_fig.title.text_font_size = font_size
    #twohalftoggles_fig.legend.visible=False
    twohalftoggles_fig.legend.background_fill_alpha=1
    twohalftoggles_fig.legend.location= "top_right"

    #
    # half toggle
    halftoggle_fig= bkplot.figure(
        frame_width=160,
        frame_height=160,
        title='tog11, tog12, tog21 mutated',
        x_axis_label="Overlapped cat RBS:ribosome \n dissoc. rate, 1/(nM*h)",
        y_axis_label="Rel. growth rate after mutation",
        x_range=(np_kplus_cats[0], np_kplus_cats[-1]),
        y_axis_type="log",
        tools="box_zoom,pan,hover,reset,save"
    )
    halftoggle_fig.output_backend = "svg"
    # Addictive toggles
    halftoggle_fig.line(np_kplus_cats,np.divide(growth_rates_halftoggle,growth_rates_bothtoggles),
                    line_color=bkRGB(222, 49, 99),line_width=2,legend_label='Overlaps')
    # Punisher
    halftoggle_fig.line(np_kplus_cats,
                    pun_growth_rate_halftoggle / pun_growth_rate_bothtoggles * np.ones(len(np_kplus_cats)),
                    line_color=bkRGB(255, 103, 0), line_width=2, legend_label='Punisher')
    # Relative growth one line
    halftoggle_fig.line(np_kplus_cats, np.ones(len(np_kplus_cats)),
                    line_color=bkRGB(0, 0, 0), line_width=2, line_dash='dashed')
    # shade region where overlaps fail - left
    halftoggle_fig.add_layout(
        bkmodels.PolyAnnotation(xs=np.concatenate((fail_kplus_cats_left, np.flip(fail_kplus_cats_left))),
                                ys=np.concatenate(
                                    (0.1 * np.ones(len(fail_kplus_cats_left)), 2 * np.ones(len(fail_kplus_cats_left)))),
                                line_width=0, line_alpha=0,
                                fill_color=bkRGB(100, 100, 100, 0.25)))
    # shade region where overlaps fail - right
    halftoggle_fig.add_layout(
        bkmodels.PolyAnnotation(xs=np.concatenate((fail_kplus_cats_right, np.flip(fail_kplus_cats_right))),
                                ys=np.concatenate((0.1 * np.ones(len(fail_kplus_cats_right)),
                                                   2 * np.ones(len(fail_kplus_cats_right)))),
                                line_width=0, line_alpha=0,
                                fill_color=bkRGB(100, 100, 100, 0.25)))

    # font and axis settings
    halftoggle_fig.xaxis.ticker = bkmodels.BasicTicker(desired_num_ticks=5)
    halftoggle_fig.xaxis.axis_label_text_font_size = font_size
    halftoggle_fig.xaxis.major_label_text_font_size = font_size
    halftoggle_fig.yaxis.axis_label_text_font_size = font_size
    halftoggle_fig.yaxis.major_label_text_font_size = font_size
    halftoggle_fig.legend.label_text_font_size = font_size
    halftoggle_fig.title.text_font_size = font_size
    #halftoggle_fig.legend.visible=False
    halftoggle_fig.legend.background_fill_alpha = 1
    halftoggle_fig.legend.location= "right"

    #
    # no toggles
    notoggles_fig= bkplot.figure(
        frame_width=160,
        frame_height=160,
        title='all toggle genes mutated',
        x_axis_label="Overlapped cat RBS:ribosome \n dissoc. rate, 1/(nM*h)",
        y_axis_label="Rel. growth rate after mutation",
        x_range=(np_kplus_cats[0], np_kplus_cats[-1]),
        y_axis_type="log",
        tools="box_zoom,pan,hover,reset,save"
    )
    notoggles_fig.output_backend = "svg"
    # Addictive toggles
    notoggles_fig.line(np_kplus_cats,np.divide(growth_rates_notoggles,growth_rates_bothtoggles),
                    line_color=bkRGB(222, 49, 99),line_width=2,legend_label='Overlaps')
    # Punisher
    notoggles_fig.line(np_kplus_cats,
                    pun_growth_rate_notoggles / pun_growth_rate_bothtoggles * np.ones(len(np_kplus_cats)),
                    line_color=bkRGB(255, 103, 0), line_width=2, legend_label='Punisher')
    # Relative growth one line
    notoggles_fig.line(np_kplus_cats, np.ones(len(np_kplus_cats)),
                    line_color=bkRGB(0, 0, 0), line_width=2, line_dash='dashed')
    # shade region where overlaps fail - left
    notoggles_fig.add_layout(
        bkmodels.PolyAnnotation(xs=np.concatenate((fail_kplus_cats_left,np.flip(fail_kplus_cats_left))),
                                ys=np.concatenate((0.1*np.ones(len(fail_kplus_cats_left)),2*np.ones(len(fail_kplus_cats_left)))),
                                line_width=0, line_alpha=0,
                                fill_color=bkRGB(100, 100, 100, 0.25)))
    # shade region where overlaps fail - right
    notoggles_fig.add_layout(
        bkmodels.PolyAnnotation(xs=np.concatenate((fail_kplus_cats_right,np.flip(fail_kplus_cats_right))),
                                ys=np.concatenate((0.1*np.ones(len(fail_kplus_cats_right)),2*np.ones(len(fail_kplus_cats_right)))),
                                line_width=0, line_alpha=0,
                                fill_color=bkRGB(100, 100, 100, 0.25)))

    # font and axis settings
    notoggles_fig.xaxis.ticker = bkmodels.BasicTicker(desired_num_ticks=5)
    notoggles_fig.xaxis.axis_label_text_font_size = font_size
    notoggles_fig.xaxis.major_label_text_font_size = font_size
    notoggles_fig.yaxis.axis_label_text_font_size = font_size
    notoggles_fig.yaxis.major_label_text_font_size = font_size
    notoggles_fig.legend.label_text_font_size = font_size
    notoggles_fig.title.text_font_size = font_size
    notoggles_fig.legend.background_fill_alpha = 1

    bkplot.save(bklayouts.grid([[toggleandhalf_fig,onetoggle_fig], [halftoggle_fig, notoggles_fig],[twohalftoggles_fig, None]]))

    return

# MAIN CALL ------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()