'''
SIMPLE_SHOWCASE.PY: A simple showcase of the punihsre reaction to synthetic gene expression loss.
'''
# By Kirill Sechkar

# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------
import numpy as np
import jax
import jax.numpy as jnp
import functools
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController, SteadyStateEvent
import pandas as pd
from bokeh import plotting as bkplot, models as bkmodels, layouts as bklayouts, io as bkio
from bokeh.colors import RGB as bkRGB

import time

# CIRCUIT AND EXTERNAL INPUT IMPORTS -----------------------------------------------------------------------------------
import synthetic_circuits_jax as circuits
from cell_model import *

# MAIN FUNCTION (FOR GENERATING THE FIGURE) ----------------------------------------------------------------------------
def main():
    # set up jax
    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update("jax_enable_x64", True)

    # initialise cell model
    cellmodel_auxil = CellModelAuxiliary()  # auxiliary tools for simulating the model and plotting simulation outcomes
    par = cellmodel_auxil.default_params()  # get default parameter values
    init_conds = cellmodel_auxil.default_init_conds(par)  # get default initial conditions

    # load synthetic gene circuit
    ode_with_circuit, circuit_F_calc, par, init_conds, circuit_genes, circuit_miscs, circuit_name2pos, circuit_styles = cellmodel_auxil.add_circuit(
        circuits.punisher_xtra_initialise,
        circuits.punisher_xtra_ode,
        circuits.punisher_xtra_F_calc,
        par, init_conds)  # load the circuit

    # burdensome gene
    par['c_xtra'] = 100.0
    par['a_xtra'] = 1000.0

    # punisher parameters
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
    init_conds['s']=0.3
    par['h_ext'] = 10.5 * (10.0 ** 3)

    # set simulation parameters
    savetimestep = 0.1  # save time step
    rtol = 1e-6  # relative tolerance for the ODE solver
    atol = 1e-6  # absolute tolerance for the ODE solver

    # initial simulation to get the steady state without gene expression loss
    tf = (0, 50)  # simulation time frame
    sol=ode_sim(par,    # dictionary with model parameters
                ode_with_circuit,   #  ODE function for the cell with synthetic circuit
                cellmodel_auxil.x0_from_init_conds(init_conds,circuit_genes,circuit_miscs),  # initial condition VECTOR
                len(circuit_genes), len(circuit_miscs), circuit_name2pos, # dictionaries with circuit gene and miscellaneous specie names, species name to vector position decoder
                cellmodel_auxil.synth_gene_params_for_jax(par,circuit_genes), # synthetic gene parameters for calculating k values
                tf, jnp.arange(tf[0], tf[1], savetimestep), # time frame and time axis for saving the system's state
                rtol, atol)    # relative and absolute tolerances
    ts=np.array(sol.ts)
    xs=np.array(sol.ys)

    # simulating synthetic gene expression loss
    par['func_xtra'] = 0.0
    tf_afterloss=(ts[-1], 200)  # simulation time frame
    x0_afterloss=sol.ys[-1,:]  # simulation will resume from the last time point
    sol_afterloss=ode_sim(par,    # dictionary with model parameters
                ode_with_circuit,   #  ODE function for the cell with synthetic circuit
                x0_afterloss,  # initial condition VECTOR
                len(circuit_genes), len(circuit_miscs), circuit_name2pos, # dictionaries with circuit gene and miscellaneous specie names, species name to vector position decoder
                cellmodel_auxil.synth_gene_params_for_jax(par,circuit_genes), # synthetic gene parameters for calculating k values
                tf_afterloss, jnp.arange(tf_afterloss[0], tf_afterloss[1], savetimestep), # time frame and time axis for saving the system's state
                rtol, atol)    # relative and absolute tolerances
    ts=np.concatenate((ts,np.array(sol_afterloss.ts)),axis=0)
    xs=np.concatenate((xs,np.array(sol_afterloss.ys)),axis=0)

    # PLOT - HOST CELL MODEL
    bkplot.output_file(filename="simpleshowcase_cellmodel.html", title="Cell Model Simulation") # set up bokeh output file
    mass_fig=cellmodel_auxil.plot_protein_masses(ts,xs,par,circuit_genes) # plot simulation results
    nat_mrna_fig,nat_prot_fig,nat_trna_fig,h_fig = cellmodel_auxil.plot_native_concentrations(ts, xs, par, circuit_genes)  # plot simulation results
    l_figure, e_figure, Fr_figure, ppGpp_figure, nu_figure, D_figure = cellmodel_auxil.plot_phys_variables(ts, xs, par, circuit_genes, circuit_miscs, circuit_name2pos)  # plot simulation results
    bkplot.save(bklayouts.grid([[mass_fig, nat_mrna_fig, nat_prot_fig],
                                [nat_trna_fig, h_fig, l_figure],
                                [e_figure, Fr_figure, D_figure]]))

    # PLOT - SYNTHETIC GENE CIRCUIT
    bkplot.output_file(filename="simpleshowcase_circuit.html", title="Synthetic Gene Circuit Simulation") # set up bokeh output file
    het_mrna_fig, het_prot_fig, misc_fig = cellmodel_auxil.plot_circuit_concentrations(ts, xs, par, circuit_genes, circuit_miscs, circuit_name2pos, circuit_styles)  # plot simulation results
    F_fig = cellmodel_auxil.plot_circuit_regulation(ts, xs, circuit_F_calc, par, circuit_genes, circuit_miscs, circuit_name2pos, circuit_styles)  # plot simulation results
    bkplot.save(bklayouts.grid([[het_mrna_fig, het_prot_fig, misc_fig],
                                [F_fig, None, None]]))

    # PLOT FIGURE 1 E - GROWTH RATE
    # get the growth rate for plotting
    _, ls, _, _, _, _, _, _ =cellmodel_auxil.get_e_l_Fr_nu_psi_T_D_Dnohet(ts, xs, par, circuit_genes, circuit_miscs, circuit_name2pos)

    # initialise
    fig1e = bkplot.figure(
        frame_width=240,
        frame_height=180,
        x_axis_label="time since mutation, h",
        y_axis_label="Cell growth rate, 1/h",
        x_range=(-5,35),
        y_range=(0.3,1.3),
        tools="box_zoom,pan,hover,reset,save"
    )
    # set svg backend
    fig1e.output_backend = "svg"

    # add a line to show when synbthetic gene expression loss occurs
    fig1e.add_layout(bkmodels.PolyAnnotation(xs=[0,0,tf_afterloss[1]-tf_afterloss[0],tf_afterloss[1]-tf_afterloss[0]],
                                             ys=[0,2,2,0],
                                             line_width=0, line_alpha=0,
                                             fill_color=bkRGB(100, 100, 100, 0.25)))
    fig1e.add_layout(bkmodels.Label(x=0, y=1.3,
                                    x_offset=2, y_offset=-16,
                                    text='Burdensome gene mutated',
                                    text_font_size='8pt'))

    # plot the growth rate
    fig1e.line(ts-tf_afterloss[0],np.array(ls), line_width=2, line_color=bkRGB(0,0,0))

    # set fonts
    fig1e.xaxis.axis_label_text_font_size = "8pt"
    fig1e.xaxis.major_label_text_font_size = "8pt"
    fig1e.yaxis.axis_label_text_font_size = "8pt"
    fig1e.yaxis.major_label_text_font_size = "8pt"

    # save the figure
    bkplot.output_file(filename="fig1e.html",title="Figure 1 e")  # set up bokeh output file
    bkplot.save(fig1e)

    # PLOT FIGURE 1 F - PROTEIN CONCENTRATIONS
    # y range for the plot (in terms of cat prot. conc.)
    fig1f_y_range = (0, 1.25 * max(np.array(xs[:, circuit_name2pos['p_cat']])))

    # initialise
    fig1f = bkplot.figure(
        frame_width=240,
        frame_height=180,
        x_axis_label="time since mutation, h",
        y_axis_label="CAT protein conc., nM",
        x_range=(-5,35),
        y_range=fig1f_y_range,
        tools="box_zoom,pan,hover,reset,save"
    )
    # set svg backend
    fig1f.output_backend = "svg"

    # add shading to show when synthetic gene expression loss occurs
    fig1f.add_layout(bkmodels.PolyAnnotation(xs=[0,0,tf_afterloss[1]-tf_afterloss[0],tf_afterloss[1]-tf_afterloss[0]],
                                             ys=[fig1f_y_range[0],fig1f_y_range[1],fig1f_y_range[1],fig1f_y_range[0]],
                                             line_width=0, line_alpha=0,
                                             fill_color=bkRGB(100, 100, 100, 0.25)))
    fig1f.add_layout(bkmodels.Label(x=0, y=fig1f_y_range[1],
                                    x_offset=2, y_offset=-16,
                                    text='Burdensome gene mutated',
                                    text_font_size='8pt'))

    # settings for the main y-axis (for p_cat)
    fig1f.yaxis.axis_line_color=bkRGB(222, 49, 99)
    fig1f.yaxis.major_tick_line_color=bkRGB(222, 49, 99)
    fig1f.yaxis.minor_tick_line_color=bkRGB(222, 49, 99)

    # plot the cat protein concentrations
    fig1f.line(ts-tf_afterloss[0],xs[:,circuit_name2pos['p_cat']], line_width=2, line_color=bkRGB(222, 49, 99), legend_label="CAT")

    # create an extra  y range for plotting integrase protein concentrations
    fig1f.extra_y_ranges = {"p_int": bkmodels.Range1d(start=0, end=1.25 * np.max(xs[:, circuit_name2pos['p_int']]))}
    fig1f.add_layout(bkmodels.LinearAxis(y_range_name="p_int",
                                         axis_label="Integrase conc., nM",
                                         axis_line_color=bkRGB(255, 103, 0),
                                         major_tick_line_color=bkRGB(255, 103, 0),
                                         minor_tick_line_color=bkRGB(255, 103, 0)),
                     'right')  # add the alternative axis label to the figure

    # plot the integrase protein concentrations
    fig1f.line(ts-tf_afterloss[0],xs[:,circuit_name2pos['p_int']], line_width=2, line_color=bkRGB(255, 103, 0), y_range_name="p_int", legend_label="Integrase")

    # add legend
    fig1f.legend.location = "right"
    fig1f.legend.label_text_font_size="8pt"

    # set fonts
    fig1f.xaxis.axis_label_text_font_size = "8pt"
    fig1f.xaxis.major_label_text_font_size = "8pt"
    fig1f.yaxis.axis_label_text_font_size = "8pt"
    fig1f.yaxis.major_label_text_font_size = "8pt"

    # save the figure
    bkplot.output_file(filename="fig1f.html",title="Figure 1 f")  # set up bokeh output file
    bkplot.save(fig1f)


    return


# MAIN CALL ------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()