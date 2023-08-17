'''
FIGA_1.PY: validation of the assumptions required to make analytical derivations
'''
# By Kirill Sechkar

# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------
import numpy as np
import jax
import jax.numpy as jnp
import functools
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController, SteadyStateEvent
import pandas as pd
import pickle
from bokeh import plotting as bkplot, models as bkmodels, layouts as bklayouts, palettes as bkpalettes, transform as bktransform
from math import pi

import time

# OWN CODE IMPORTS -----------------------------------------------------------------------------------------------------
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import synthetic_circuits_jax as circuits
from cell_model import *

# ODE SIMULATION WITH VMAPPING -----------------------------------------------------------------------------------------
# get steady states for given initial conditions and parameters (vmapped)
def get_steadystates(par,    # dictionary with model parameters
                        ode_with_circuit,  # ODE function for the cell with the synthetic gene circuit
                        x0,  # initial condition vector
                        num_circuit_genes, num_circuit_miscs, circuit_name2pos, sgp4j, # dictionaries with circuit gene and miscellaneous specie names, species name to vector position decoder, relevant synthetic gene parameters in jax.array form
                        tf, rtol, atol    # simulation parameters: time frame, when to save the system's state, relative and absolute tolerances
                        ):
    # define the ODE term
    vector_field = lambda t, y, args: ode_with_circuit(t, y, args)
    term = ODETerm(vector_field)

    # define arguments of the ODE term
    args = (
        par,  # model parameters
        circuit_name2pos,  # gene name - position in circuit vector decoder
        num_circuit_genes, num_circuit_miscs,  # number of genes and miscellaneous species in the circuit
        sgp4j  # relevant synthetic gene parameters in jax.array form
    )

    # define the solver
    solver = Dopri5()

    # define the timestep controller
    stepsize_controller = PIDController(rtol=rtol, atol=atol)

    # define the steady-state termination conditions
    steady_state_stop = SteadyStateEvent(rtol=0.001,atol=0.001)  # stop simulation prematurely if steady state is reached

    # solvew the ODE
    sol = diffeqsolve(term, solver,
                      args=args,
                      t0=tf[0], t1=tf[1], dt0=0.1, y0=x0,
                      max_steps=None,
                      stepsize_controller=stepsize_controller)

    return sol


# PLOT PERCENTAGE CHANGES RELATIVE TO NO BURDEN ------------------------------------------------------------------------
# get ABSOLUTE percentage changes relative to no burden for a given variable
def get_percentage_changes(var):
    noburden_baselines=np.multiply(np.ones(var.shape),np.atleast_2d(var[:, 0]).T)
    return np.abs((var - noburden_baselines) / noburden_baselines * 100)

# plot the heatmap of changes relative to no burden
def plot_heatmap(var_changes,  # physiological variable to be plotted
                 var_description, # description of the variable
                 nutr_qual_range_jnp,  # range of nutrient qualities considered (jnp array)
                 c_xtras_jnp,  # range of heterologous gene expression rates considered (jnp array)
                 colourbar_range = (None,),  # range of the colour scale
                 plot_colourbar=True,  # whether to plot the colourbar
                 threshold_percentage=None,  # threshold percentage to mark on the heatmap
                 dimensions=(480,420)  # dimensions of the plot (width, height)
                 ):
    # make sure nutrient qualities and gene concentration are numpy arrays
    nutr_qual_range = np.round(np.array(nutr_qual_range_jnp),2)
    c_xtras = np.round(np.array(c_xtras_jnp),2)

    # make axis labels
    nutr_qual_labels = []
    for i in range(0, len(nutr_qual_range)):
        if(i==0 or i==int(len(nutr_qual_range+1)/2) or i==len(nutr_qual_range)-1):
            nutr_qual_labels.append(str(nutr_qual_range[i]))
        else:
            nutr_qual_labels.append('')
    c_xtras_labels = []
    for i in range(0, len(c_xtras)):
        if(i==0 or i==int(len(c_xtras+1)/2) or i==len(c_xtras)-1):
            c_xtras_labels.append(str(c_xtras[i]))
        else:
            c_xtras_labels.append('')

    # make a dataframe for plotting
    str_c_xtras=[np.format_float_scientific(x, precision=2) for x in c_xtras]
    df_2d = pd.DataFrame(var_changes, columns=str_c_xtras)
    df_2d.columns = df_2d.columns.astype('str')
    df_2d['Nutrient quality'] = nutr_qual_range
    df_2d['Nutrient quality'] = df_2d['Nutrient quality'].astype('str')
    df_2d = df_2d.set_index('Nutrient quality')
    df_2d.columns.name = 'Gene xtra promoter strength'
    es_df = pd.DataFrame(df_2d.stack(), columns=[var_description]).reset_index()

    # set up the colour bar range (if not specified)
    if(colourbar_range[0]==None):
        colourbar_range = (min(es_df[var_description]), max(es_df[var_description]))

    # set up the graph
    figure = bkplot.figure(
        x_axis_label='Nutrient quality',
        y_axis_label='Gene xtra promoter strength',
        x_range=list(df_2d.index),
        y_range=list(df_2d.columns),
        width=dimensions[0], height=dimensions[1],
        tools="box_zoom,pan,hover,reset,save",
        tooltips=[('Nutr. qual. = ','@{Nutrient quality}'),
                  ('Gene conc. = ','@{Gene xtra promoter strength}'),
                  (var_description+' change=', '@'+var_description)],
        title=var_description+' change from no synth. gene exp. burden',
    )
    # svg backend
    figure.output_backend = "svg"
    figure.grid.grid_line_color = None
    figure.axis.axis_line_color = None
    figure.axis.major_tick_line_color = None
    figure.axis.major_label_text_font_size = "8pt"
    figure.axis.major_label_standoff = 0
    figure.xaxis.major_label_orientation = pi/2

    # plot the heatmap
    rects = figure.rect(x="Nutrient quality", y="Gene xtra promoter strength", source=es_df,
                       width=1, height=1,
                       fill_color=bktransform.linear_cmap(var_description, bkpalettes.Turbo256, low=colourbar_range[0],high=colourbar_range[1]),
                       line_color=None)
    # add colour bar
    if(plot_colourbar):
        figure.add_layout(rects.construct_color_bar(
            major_label_text_font_size="8pt",
            ticker=bkmodels.BasicTicker(desired_num_ticks=3),
            formatter=bkmodels.PrintfTickFormatter(format="%.3f%%"),
            label_standoff=6,
            border_line_color=None,
            padding=5
        ), 'right')

    # mark the percentage threshold on the heatmap
    if(threshold_percentage!=None):
        max_change_for_nutr_qual = np.max(var_changes, axis=1)
        for i in range(0, len(nutr_qual_range)):
            if(max_change_for_nutr_qual[i]<=threshold_percentage):
                figure.add_layout(bkmodels.Span(location=i, dimension='height', line_color='white', line_width=5))
                break

    return figure

# MAIN BODY ------------------------------------------------------------------------------------------------------------
def main():
    # PREPARE ----------------------------------------------------------------------------------------------------------
    # set up jax
    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update("jax_enable_x64", True)
    import os, multiprocessing
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(multiprocessing.cpu_count())

    # set up the circuit
    cellmodel_auxil = CellModelAuxiliary()  # auxiliary tools for simulating the model and plotting simulation outcomes
    par = cellmodel_auxil.default_params()  # get default parameter values
    init_conds = cellmodel_auxil.default_init_conds(par)  # get default initial conditions
    # ode_with_circuit, circuit_F_calc, par, init_conds, circuit_genes, circuit_miscs, circuit_name2pos, circuit_colours = cellmodel_auxil.add_circuit(
    #     circuits.oneconstitutive_initialise,
    #     circuits.oneconstitutive_ode,
    #     circuits.oneconstitutive_F_calc,
    #     par, init_conds)  # load the circuit (WITHOUT CAT - comment out the unused one)
    ode_with_circuit, circuit_F_calc, par, init_conds, circuit_genes, circuit_miscs, circuit_name2pos, circuit_colours = cellmodel_auxil.add_circuit(
        circuits.oneconstitutive_cat_initialise,
        circuits.oneconstitutive_cat_ode,
        circuits.oneconstitutive_cat_F_calc,
        par, init_conds)  # load the circuit (WITH CAT - comment out the unused one)

    # define range of heterologous gene expression rates
    num_c_xtras = 20
    par['a_xtra']=1
    par['c_xtra']= jnp.linspace(0,par['a_a']*2/par['a_xtra'],num_c_xtras)    # plasmid concentration range

    # cat gene parameters and chloramphenicol level - if simulating the circuit with cat
    if('cat' in circuit_genes):
        print('Running with cat')
        par['c_cat'] = 1  # gene concentration (nM)
        par['a_cat'] = 5000  # promoter strength (unitless)
        par['h_ext'] = 10.5 * (10 ** 3)  # chloramphenicol concentration

    # define range of nutrient qualities
    num_nutr_quals = 20
    nutr_qual_range = jnp.linspace(0.05, 1, num_nutr_quals)  # nutrient quality range
    x0_default = cellmodel_auxil.x0_from_init_conds(init_conds,circuit_genes,circuit_miscs)  # get default x0 value
    x0s_unswapped = jnp.multiply(np.ones((len(nutr_qual_range), len(x0_default))), x0_default)
    x0s = x0s_unswapped.at[:, 6].set(nutr_qual_range[:])  # set s values in x0s

    # vmap axes for parameters
    par_vmap_axes = {}
    for parameter in par.keys():
        if(parameter=='c_xtra'):
            par_vmap_axes[parameter]=0
        else:
            par_vmap_axes[parameter]=None

    # define simulation parameters
    tf = (0, 48) # simulation time frame - assume that the cell is close to steady state after 48h
    rtol = 1e-6; atol = 1e-6  # relative and absolute tolerances for the ODE solver

    # initialise output dictionaries (to make dataframes)
    es = np.zeros((len(nutr_qual_range),len(par['c_xtra'])))
    Ts = np.zeros((len(nutr_qual_range),len(par['c_xtra'])))
    F_rs = np.zeros((len(nutr_qual_range),len(par['c_xtra'])))
    nus = np.zeros((len(nutr_qual_range),len(par['c_xtra'])))

    # SIMULATE ---------------------------------------------------------------------------------------------------------
    simulate = False # simulate from scratch or load a saved run from a pickle file
    if(simulate):
        print('Simulating...')
        all_timer = time.time()  # start timer
        sgp4j=cellmodel_auxil.synth_gene_params_for_jax(par,circuit_genes) # pre-caluclate jaxed parameters for finding k vlaues for synthetic genes
        for i in range(0,len(nutr_qual_range)):
            one_run_timer = time.time()  # start timer
            get_steadystates_for_par = lambda par: get_steadystates(par,  # dictionary with model parameters
                                                                    ode_with_circuit,  # ODE function for the cell with synthetic circuit
                                                                    x0s[i,:],  # initial condition vectorS
                                                                    len(circuit_genes), len(circuit_miscs), circuit_name2pos,
                                                                    # dictionaries with circuit gene and miscellaneous specie names, species name to vector position decoder
                                                                    sgp4j,# synthetic gene parameters for calculating k values
                                                                    tf,  # maximum simulation time frame
                                                                    rtol,atol)  # simulation parameters: when to save the system's state, relative and absolute tolerances)   # simulation parameters: time frame, save time step, relative and absolute tolerances
            vmapped_get_steadystates_for_par = jax.pmap(get_steadystates_for_par, in_axes=(par_vmap_axes,))
            sols = vmapped_get_steadystates_for_par(par)
            for j in range(0,len(par['c_xtra'])):
                # get steady state values of relevant physiological variables
                par_with_single_c_xtra = par.copy()
                par_with_single_c_xtra['c_xtra']=par['c_xtra'][j]
                e, l, F_r, nu, _, T, D, D_nohet = cellmodel_auxil.get_e_l_Fr_nu_psi_T_D_Dnohet(sols.ts[j,0],sols.ys[j,:,:],par_with_single_c_xtra,
                                                                                               circuit_genes,circuit_miscs,circuit_name2pos)
                # record the obtained values
                es[i,j] = e[0]
                Ts[i,j] = T[0]
                F_rs[i,j] = F_r[0]
                nus[i,j] = nu[0]
            print('Completed for nutrient quaility '+str(i)+' of '+str(len(nutr_qual_range))+' in '+str(time.time()-one_run_timer)+' seconds')
        print('Simulations completed in '+ str(time.time()-all_timer) + ' seconds')
        # pickle the outcome
        pickle.dump((es,Ts,F_rs,nus),open('validation runs/assumption_validation_' + str(len(nutr_qual_range)) + 'x' + str(len(par['c_xtra'])) + '.pkl','wb'))
    else:
        print('Loading saved output...')
        # load the outcome from pickle files
        es,Ts,F_rs,nus = pickle.load(open('validation runs/assumption_validation_' + str(len(nutr_qual_range)) + 'x' + str(len(par['c_xtra'])) + '.pkl','rb'))

    # PLOT -------------------------------------------------------------------------------------------------------------
    # open file
    bkplot.output_file('assumption_validation.html',title='Assumption validation')

    # find percentage changes relative to no burden
    es_changes = get_percentage_changes(es)
    Ts_changes = get_percentage_changes(Ts)
    F_rs_changes = get_percentage_changes(F_rs)
    nus_changes = get_percentage_changes(nus)

    # get a unified colourbar range
    min_val = 0
    max_val = max([np.max(es_changes),np.max(Ts_changes),np.max(F_rs_changes),np.max(nus_changes)])

    # plot the heatmaps
    e_fig = plot_heatmap(es_changes, 'ϵ', nutr_qual_range, par['c_xtra'],
                         threshold_percentage=10,
                         colourbar_range=(min_val,max_val))
    T_fig = plot_heatmap(Ts_changes, 'T', nutr_qual_range, par['c_xtra'],
                         threshold_percentage=10,
                         colourbar_range=(min_val,max_val))
    F_r_fig = plot_heatmap(F_rs_changes, 'F_r', nutr_qual_range, par['c_xtra'],
                         threshold_percentage=10,
                         colourbar_range=(min_val,max_val))
    nu_fig = plot_heatmap(nus_changes, 'nu', nutr_qual_range, par['c_xtra'],
                         threshold_percentage=10,
                         colourbar_range=(min_val,max_val))

    # save plot
    bkplot.save(bklayouts.grid([[e_fig, T_fig],
                                [F_r_fig, nu_fig]]))
    return

# MAIN CALL ------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()