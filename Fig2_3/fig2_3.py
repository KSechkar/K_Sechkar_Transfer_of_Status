'''
FIG2_3.PY - Consider the case of a punisher circuit with two toggle switches.
Also includes code for simulating gene overlaps with two toggle switches.
Can also be amended to obtain figure A2.
'''
# By Kirill Sechkar

# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------
import numpy as np
import jax
import jax.numpy as jnp
import functools
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController, SteadyStateEvent
import pandas as pd
from bokeh import plotting as bkplot, models as bkmodels, layouts as bklayouts
from bokeh.colors import RGB as bkRGB
from math import pi

import time

# CIRCUIT AND EXTERNAL INPUT IMPORTS -----------------------------------------------------------------------------------
import synthetic_circuits_jax as circuits
from cell_model import *

# GENERATE FIGURE FOR THE LOSS OF ONE TOGGLE SWITCH --------------------------------------------------------------------
def lose_one_toggle(case_id,  # short case identifier for the output file name
                    par,  # dictionary with model parameters
                    ode_with_circuit,  # ODE function for the cell with synthetic circuit
                    circuit_F_calc,  # function for calculating the circuit's reaction rates
                    init_conds,  # initial condition DICTIONARY
                    circuit_genes, circuit_miscs,  # dictionaries with circuit gene and miscellaneous specie names
                    circuit_name2pos,  # species name to vector position decoder for the circuit
                    circuit_styles,  # dictionary with circuit gene and miscellaneous specie plotting styles
                    savetimestep, rtol, atol  # time step for saving the simulation, relative and absolute tolerances
                    ):
    # get an auxiliary tool for simulating the model and plotting simulation outcomes
    cellmodel_auxil = CellModelAuxiliary()

    # initial simulation to get the steady state without gene expression loss
    tf = (0, 50)  # simulation time frame
    sol = ode_sim(par,  # dictionary with model parameters
                  ode_with_circuit,  # ODE function for the cell with synthetic circuit
                  cellmodel_auxil.x0_from_init_conds(init_conds, circuit_genes, circuit_miscs),  # initial condition VECTOR
                  len(circuit_genes), len(circuit_miscs), circuit_name2pos,  # dictionaries with circuit gene and miscellaneous specie names, species name to vector position decoder
                  cellmodel_auxil.synth_gene_params_for_jax(par, circuit_genes), # synthetic gene parameters for calculating k values
                  tf, jnp.arange(tf[0], tf[1], savetimestep),  # time frame and time axis for saving the system's state
                  rtol, atol)  # relative and

    # absolute tolerances
    ts = np.array(sol.ts)
    xs = np.array(sol.ys)

    # simulating synthetic gene expression loss
    par['func_tog11'] = 0.0
    par['func_tog12'] = 0.0
    tf_afterloss = (ts[-1], 85)  # simulation time frame
    x0_afterloss = sol.ys[-1, :]  # simulation will resume from the last time point
    sol_afterloss = ode_sim(par,  # dictionary with model parameters
                            ode_with_circuit,  # ODE function for the cell with synthetic circuit
                            x0_afterloss,  # initial condition VECTOR
                            len(circuit_genes), len(circuit_miscs), circuit_name2pos,
                            # dictionaries with circuit gene and miscellaneous specie names, species name to vector position decoder
                            cellmodel_auxil.synth_gene_params_for_jax(par, circuit_genes),
                            # synthetic gene parameters for calculating k values
                            tf_afterloss, jnp.arange(tf_afterloss[0], tf_afterloss[1], savetimestep),
                            # time frame and time axis for saving the system's state
                            rtol, atol)  # relative and absolute tolerances
    ts = np.concatenate((ts, np.array(sol_afterloss.ts)), axis=0)
    xs = np.concatenate((xs, np.array(sol_afterloss.ys)), axis=0)

    # plotting the simulation results
    # PLOT - HOST CELL MODEL
    bkplot.output_file(filename=case_id+"_loseonetoggle_cellmodel.html",
                       title="Cell Model Simulation")  # set up bokeh output file
    mass_fig = cellmodel_auxil.plot_protein_masses(ts, xs, par, circuit_genes)  # plot simulation results
    nat_mrna_fig, nat_prot_fig, nat_trna_fig, h_fig = cellmodel_auxil.plot_native_concentrations(ts, xs, par,
                                                                                                 circuit_genes)  # plot simulation results
    l_figure, e_figure, Fr_figure, ppGpp_figure, nu_figure, D_figure = cellmodel_auxil.plot_phys_variables(ts, xs, par,
                                                                                                           circuit_genes,
                                                                                                           circuit_miscs,
                                                                                                           circuit_name2pos)  # plot simulation results
    bkplot.save(bklayouts.grid([[mass_fig, nat_mrna_fig, nat_prot_fig],
                                [nat_trna_fig, h_fig, l_figure],
                                [e_figure, Fr_figure, D_figure]]))

    # PLOT - SYNTHETIC GENE CIRCUIT
    bkplot.output_file(filename=case_id+"_loseonetoggle_circuit.html",
                       title="Synthetic Gene Circuit Simulation")  # set up bokeh output file
    het_mrna_fig, het_prot_fig, misc_fig = cellmodel_auxil.plot_circuit_concentrations(ts, xs, par, circuit_genes,
                                                                                       circuit_miscs, circuit_name2pos,
                                                                                       circuit_styles)  # plot simulation results
    F_fig = cellmodel_auxil.plot_circuit_regulation(ts, xs, circuit_F_calc, par, circuit_genes, circuit_miscs,
                                                    circuit_name2pos, circuit_styles)  # plot simulation results
    bkplot.save(bklayouts.grid([[het_mrna_fig, het_prot_fig, misc_fig],
                                [F_fig, None, None]]))

    # PLOT PAPER FIGURES -----------------------------------------------------------------------------------------------
    # PLOT FIGURE 3B - TOGGLE PROTEIN CONCENTRATIONS
    # initialise figure
    prot_fig = bkplot.figure(
        frame_width=240,
        frame_height=180,
        x_axis_label="time since mutation, h",
        y_axis_label="protein conc., nM",
        x_range=(-20, tf_afterloss[1]-tf_afterloss[0]),
        y_range=(0,1e5),
        tools="box_zoom,pan,hover,reset,save"
    )
    # set svg backend
    prot_fig.output_backend = "svg"

    # add shading to show when synthetic gene expression loss occurs
    prot_fig.add_layout(bkmodels.PolyAnnotation(xs=[0, 0, tf_afterloss[1]-tf_afterloss[0], tf_afterloss[1]-tf_afterloss[0]],
                                             ys=[0, 1.3e5, 1.3e5, 0],
                                             line_width=0, line_alpha=0,
                                             fill_color=bkRGB(100, 100, 100, 0.25)))
    prot_fig.add_layout(bkmodels.Label(x=0, y=1e5,
                                    x_offset=2, y_offset=-16,
                                    text='Toggle 1 genes mutated',
                                    text_font_size='8pt'))

    # plot the toggle switch protein concentrations for tog11 and tog21
    prot_fig.line(ts-tf_afterloss[0], xs[:, circuit_name2pos['p_tog11']], line_width=2, line_color=bkRGB(100, 149, 237), legend_label='tog11')
    prot_fig.line(ts-tf_afterloss[0], xs[:, circuit_name2pos['p_tog21']], line_width=2, line_color=bkRGB(207, 181, 59), legend_label='tog21')

    # plot the toggle switch protein concentrations for tog12 and tog22
    prot_fig.line(ts - tf_afterloss[0], xs[:, circuit_name2pos['p_tog12']], line_width=2, line_color=bkRGB(127, 255, 212),
               legend_label='tog12')
    prot_fig.line(ts - tf_afterloss[0], xs[:, circuit_name2pos['p_tog22']], line_width=2, line_color=bkRGB(252, 194, 0),
               legend_label='tog22')
    
    # legend formatting
    prot_fig.legend.location = "bottom_left"
    prot_fig.legend.label_text_font_size = "8pt"

    # set fonts
    prot_fig.xaxis.axis_label_text_font_size = "8pt"
    prot_fig.xaxis.major_label_text_font_size = "8pt"
    prot_fig.yaxis.axis_label_text_font_size = "8pt"
    prot_fig.yaxis.major_label_text_font_size = "8pt"
    prot_fig.yaxis[0].formatter = bkmodels.PrintfTickFormatter(format="%.0e")

    # save the figure
    bkplot.output_file(filename=case_id+"prot_fig.html", title="Figure 3 b")  # set up bokeh output file
    bkplot.save(prot_fig)

    # PLOT FIGURE 3C - INTEGRASE CONCENTRATIONS AND GROWTH RATE
    # get the growth rate for plotting
    _, ls, _, _, _, _, _, _ = cellmodel_auxil.get_e_l_Fr_nu_psi_T_D_Dnohet(ts, xs, par, circuit_genes, circuit_miscs,
                                                                           circuit_name2pos)

    # initialise
    growth_fig = bkplot.figure(
        frame_width=240,
        frame_height=180,
        x_axis_label="time since mutation, h",
        y_axis_label="Cell growth rate, 1/h",
        x_range=(-20, tf_afterloss[1]-tf_afterloss[0]),
        y_range=(0.3, 1.3),
        tools="box_zoom,pan,hover,reset,save"
    )
    # set svg backend
    growth_fig.output_backend = "svg"

    # add a line to show when synbthetic gene expression loss occurs
    growth_fig.add_layout(
        bkmodels.PolyAnnotation(xs=[0, 0, tf_afterloss[1] - tf_afterloss[0], tf_afterloss[1] - tf_afterloss[0]],
                                ys=[0, 2, 2, 0],
                                line_width=0, line_alpha=0,
                                fill_color=bkRGB(100, 100, 100, 0.25)))
    growth_fig.add_layout(bkmodels.Label(x=0, y=1.3,
                                    x_offset=2, y_offset=-16,
                                    text='Toggle 1 genes mutated',
                                    text_font_size='8pt'))

    # plot the growth rate
    growth_fig.line(ts - tf_afterloss[0], np.array(ls), line_width=2, line_color=bkRGB(0, 0, 0), legend_label='Growth rate')

    # for the punisher, we also plot integrase levels
    if(case_id=='pun'):
        # create an extra  y range for plotting integrase protein concentrations
        growth_fig.extra_y_ranges = {"p_int": bkmodels.Range1d(start=0, end=250)}
        growth_fig.add_layout(bkmodels.LinearAxis(y_range_name="p_int",
                                             axis_label="Integrase conc., nM",
                                             axis_line_color=bkRGB(255, 103, 0),
                                             major_tick_line_color=bkRGB(255, 103, 0),
                                             minor_tick_line_color=bkRGB(255, 103, 0)),
                         'right')  # add the alternative axis label to the figure

        # plot the integrase protein concentrations
        growth_fig.line(ts - tf_afterloss[0], xs[:, circuit_name2pos['p_int']], line_width=2, line_color=bkRGB(255, 103, 0),
                   y_range_name="p_int", legend_label="Int. conc.")

    # set fonts
    growth_fig.xaxis.axis_label_text_font_size = "8pt"
    growth_fig.xaxis.major_label_text_font_size = "8pt"
    growth_fig.yaxis.axis_label_text_font_size = "8pt"
    growth_fig.yaxis.major_label_text_font_size = "8pt"

    # legend formatting
    growth_fig.legend.location = "left"
    growth_fig.legend.label_text_font_size = "8pt"

    # save the figure
    bkplot.output_file(filename=case_id+"growth_fig.html", title="Figure 3 c")  # set up bokeh output file
    bkplot.save(growth_fig)

    # for addiction, we also plot Cm and CAT levels
    if(case_id == 'add'):
        # get total cat levels
        cats = xs[:, circuit_name2pos['p_cat11']] + xs[:, circuit_name2pos['p_cat12']] + xs[:, circuit_name2pos['p_cat21']] + xs[:, circuit_name2pos['p_cat22']]

        # get y-axis range for CAT concs.
        cat_yrange = (0, 1.25 * np.max(cats))

        cat_fig = bkplot.figure(
            frame_width=240,
            frame_height=180,
            x_axis_label="time since mutation, h",
            y_axis_label="CAT concentration, nM",
            x_range=(-5, tf_afterloss[1] - tf_afterloss[0]),
            y_range=cat_yrange,
            tools="box_zoom,pan,hover,reset,save"
        )
        # set svg backend
        growth_fig.output_backend = "svg"

        # plot CAT concs.
        cat_fig.line(ts - tf_afterloss[0], cats, line_width=2, line_color=bkRGB(222, 49, 99), legend_label='CAT')

        # add a line to show when synthetic gene expression loss occurs
        cat_fig.add_layout(
            bkmodels.PolyAnnotation(xs=[0, 0, tf_afterloss[1] - tf_afterloss[0], tf_afterloss[1] - tf_afterloss[0]],
                                    ys=[0, 2, 2, 0],
                                    line_width=0, line_alpha=0,
                                    fill_color=bkRGB(100, 100, 100, 0.25)))
        cat_fig.add_layout(bkmodels.Label(x=0, y=1.5,
                                             x_offset=2, y_offset=-16,
                                             text='Toggle 1 genes mutated',
                                             text_font_size='8pt'))

        # plot CAT concs.
        cat_fig.line(ts - tf_afterloss[0], cats, line_width=2, line_color=bkRGB(0, 0, 0), legend_label='CAT')

        # create an extra  y range for plotting integrase protein concentrations
        cat_fig.extra_y_ranges = {"Cm": bkmodels.Range1d(start=0, end=1.25 * np.max(xs[:, 7]))}
        cat_fig.add_layout(bkmodels.LinearAxis(y_range_name="Cm",
                                                  axis_label="Choramphenicol conc., nM",
                                                  axis_line_color=bkRGB(252,194,0),
                                                  major_tick_line_color=bkRGB(252,194,0),
                                                  minor_tick_line_color=bkRGB(252,194,0)),
                              'right')  # add the alternative axis label to the figure

        # plot Cm concentrations
        cat_fig.line(ts - tf_afterloss[0], xs[:, 7], line_width=2,
                        line_color=bkRGB(252,194,0),
                        y_range_name="Cm", legend_label="Cm")

        # settings for the main y-axis (for p_cat)
        cat_fig.yaxis.axis_line_color = bkRGB(222, 49, 99)
        cat_fig.yaxis.major_tick_line_color = bkRGB(222, 49, 99)
        cat_fig.yaxis.minor_tick_line_color = bkRGB(222, 49, 99)

        # legend settings
        cat_fig.legend.location = "bottom_left"
        cat_fig.legend.label_text_font_size = "8pt"

        # set fonts
        cat_fig.xaxis.axis_label_text_font_size = "8pt"
        cat_fig.xaxis.major_label_text_font_size = "8pt"
        cat_fig.yaxis.axis_label_text_font_size = "8pt"
        cat_fig.yaxis.major_label_text_font_size = "8pt"

        # save plot
        bkplot.output_file(filename=case_id+"_cat_fig.html", title="Figure 4d")  # set up bokeh output file
        bkplot.save(cat_fig)


    return


# GENERATE FIGURE FOR FLIPPING ONE TOGGLE SWITCH -----------------------------------------------------------------------
def flip_toggles(case_id,  # short case identifier for the output file name
                 par,  # dictionary with model parameters
                 ode_with_circuit,  # ODE function for the cell with synthetic circuit
                 circuit_F_calc,  # function for calculating the circuit's reaction rates
                 init_conds,  # initial condition DICTIONARY
                 circuit_genes, circuit_miscs,  # dictionaries with circuit gene and miscellaneous specie names
                 circuit_name2pos,  # species name to vector position decoder for the circuit
                 circuit_styles,  # dictionary with circuit gene and miscellaneous specie plotting styles
                 savetimestep, rtol, atol  # time step for saving the simulation, relative and absolute tolerances
                 ):
    # get an auxiliary tool for simulating the model and plotting simulation outcomes
    cellmodel_auxil = CellModelAuxiliary()

    # initial simulation to get the steady state without gene expression loss

    tf = (0, 50)  # simulation time frame
    sol = ode_sim(par,  # dictionary with model parameters
                  ode_with_circuit,  # ODE function for the cell with synthetic circuit
                  cellmodel_auxil.x0_from_init_conds(init_conds, circuit_genes, circuit_miscs),  # initial condition VECTOR
                  len(circuit_genes), len(circuit_miscs), circuit_name2pos,  # dictionaries with circuit gene and miscellaneous specie names, species name to vector position decoder
                  cellmodel_auxil.synth_gene_params_for_jax(par, circuit_genes), # synthetic gene parameters for calculating k values
                  tf, jnp.arange(tf[0], tf[1], savetimestep),  # time frame and time axis for saving the system's state
                  rtol, atol)  # relative and absolute tolerances
    ts = np.array(sol.ts)
    xs = np.array(sol.ys)

    # simulate flipping of one toggle switch by manipulating the inducer concentration
    pulse_time = 1 # time over which the inducer concentration for one toggle switch is pulsed
    pulse_ptog11_acfrac = 0.0 # active fraction of p_tog11 during the inducer pulse
    pulse_ptog21_acfrac = 0.0 # active fraction of p_tog21 during the inducer pulse
    orig_ptog11_acfrac = par['p_tog11_ac_frac']  # save the original active fraction of p_tog11
    orig_ptog21_acfrac = par['p_tog21_ac_frac']  # save the original active fraction of p_tog21
    par['p_tog11_ac_frac'] = pulse_ptog11_acfrac  # set the active fraction of p_tog11 to the pulse value
    par['p_tog21_ac_frac'] = pulse_ptog21_acfrac  # set the active fraction of p_tog21 to the pulse value
    tf_pulse = (ts[-1], ts[-1] + pulse_time)  # simulation time frame
    x0_pulse = sol.ys[-1, :]  # simulation will resume from the last time point
    sol_pulse = ode_sim(par,  # dictionary with model parameters
                            ode_with_circuit,  # ODE function for the cell with synthetic circuit
                            x0_pulse,  # initial condition VECTOR
                            len(circuit_genes), len(circuit_miscs), circuit_name2pos,  # dictionaries with circuit gene and miscellaneous specie names, species name to vector position decoder
                            cellmodel_auxil.synth_gene_params_for_jax(par, circuit_genes), # synthetic gene parameters for calculating k values
                            tf_pulse, jnp.arange(tf_pulse[0], tf_pulse[1], savetimestep),  # time frame and time axis for saving the system's state
                            rtol, atol)  # relative and absolute tolerances
    ts = np.concatenate((ts, np.array(sol_pulse.ts)), axis=0)
    xs = np.concatenate((xs, np.array(sol_pulse.ys)), axis=0)

    # after the pulse, inducer levels return to the original state
    par['p_tog11_ac_frac'] = orig_ptog11_acfrac  # set the active fraction of p_tog11 to the pulse value
    par['p_tog21_ac_frac'] = orig_ptog21_acfrac  # set the active fraction of p_tog21 to the pulse value
    tf_afterpulse = (ts[-1], 70)  # simulation time frame
    x0_afterpulse = sol_pulse.ys[-1, :]  # simulation will resume from the last time point
    sol_afterpulse = ode_sim(par,  # dictionary with model parameters
                            ode_with_circuit,  # ODE function for the cell with synthetic circuit
                            x0_afterpulse,  # initial condition VECTOR
                            len(circuit_genes), len(circuit_miscs), circuit_name2pos,  # dictionaries with circuit gene and miscellaneous specie names, species name to vector position decoder
                            cellmodel_auxil.synth_gene_params_for_jax(par, circuit_genes),  # synthetic gene parameters for calculating k values
                            tf_afterpulse, jnp.arange(tf_afterpulse[0], tf_afterpulse[1], savetimestep),  # time frame and time axis for saving the system's state
                            rtol, atol)  # relative and absolute tolerances
    ts = np.concatenate((ts, np.array(sol_afterpulse.ts)), axis=0)
    xs = np.concatenate((xs, np.array(sol_afterpulse.ys)), axis=0)

    # plotting the simulation results
    # PLOT - HOST CELL MODEL
    bkplot.output_file(filename=case_id+"_fliptoggles_cellmodel.html",
                       title="Cell Model Simulation")  # set up bokeh output file
    mass_fig = cellmodel_auxil.plot_protein_masses(ts, xs, par, circuit_genes)  # plot simulation results
    nat_mrna_fig, nat_prot_fig, nat_trna_fig, h_fig = cellmodel_auxil.plot_native_concentrations(ts, xs, par,
                                                                                                 circuit_genes)  # plot simulation results
    l_figure, e_figure, Fr_figure, ppGpp_figure, nu_figure, D_figure = cellmodel_auxil.plot_phys_variables(ts, xs, par,
                                                                                                           circuit_genes,
                                                                                                           circuit_miscs,
                                                                                                           circuit_name2pos)  # plot simulation results
    bkplot.save(bklayouts.grid([[mass_fig, nat_mrna_fig, nat_prot_fig],
                                [nat_trna_fig, h_fig, l_figure],
                                [e_figure, Fr_figure, D_figure]]))

    # PLOT - SYNTHETIC GENE CIRCUIT
    bkplot.output_file(filename=case_id+"_fliptoggles_circuit.html",
                       title="Synthetic Gene Circuit Simulation")  # set up bokeh output file
    het_mrna_fig, het_prot_fig, misc_fig = cellmodel_auxil.plot_circuit_concentrations(ts, xs, par, circuit_genes,
                                                                                       circuit_miscs, circuit_name2pos,
                                                                                       circuit_styles)  # plot simulation results


    F_fig = cellmodel_auxil.plot_circuit_regulation(np.array(sol.ts), np.array(sol.ys), circuit_F_calc, par, circuit_genes, circuit_miscs,
                                                    circuit_name2pos, circuit_styles)  # plot simulation results
    F_fig_afterpulse = cellmodel_auxil.plot_circuit_regulation(np.array(sol_afterpulse.ts), np.array(sol_afterpulse.ys), circuit_F_calc, par, circuit_genes, circuit_miscs,
                                                    circuit_name2pos, circuit_styles)  # plot simulation results
    par['p_tog_11_ac_frac'] = pulse_ptog11_acfrac  # set the active fraction of p_tog11 to the pulse value
    F_fig_pulse = cellmodel_auxil.plot_circuit_regulation(np.array(sol_pulse.ts), np.array(sol_pulse.ys), circuit_F_calc, par, circuit_genes, circuit_miscs,
                                                    circuit_name2pos, circuit_styles)  # plot simulation results
    bkplot.save(bklayouts.grid([[het_mrna_fig, het_prot_fig, misc_fig],
                                [F_fig, F_fig_pulse, F_fig_afterpulse]]))

    # PLOT PAPER FIGURES -----------------------------------------------------------------------------------------------
    # PLOT FIGURE 3D - TOGGLE PROTEIN CONCENTRATIONS
    # initialise figure
    fig3d = bkplot.figure(
        frame_width=240,
        frame_height=180,
        x_axis_label="time since pulse start, h",
        y_axis_label="protein conc., nM",
        x_range=(-5, tf_afterpulse[1] - tf_pulse[0]),
        y_range=(0,1e5),
        tools="box_zoom,pan,hover,reset,save"
    )
    # set svg backend
    fig3d.output_backend = "svg"

    # add shading to show when synthetic gene expression loss occurs
    fig3d.add_layout(
        bkmodels.PolyAnnotation(xs=[0, 0, tf_pulse[1] - tf_pulse[0], tf_pulse[1] - tf_pulse[0]],
                                ys=[0, 1e5, 1e5, 0],
                                line_width=0, line_alpha=0,
                                fill_color=bkRGB(100, 100, 100, 0.25)))
    fig3d.add_layout(bkmodels.Label(x=0, y=1e5,
                                    x_offset=0, y_offset=-145,
                                    text='Toggle 12 & 22 inducer pulse',
                                    text_font_size='8pt',
                                    angle=pi/2))

    # plot the toggle switch protein concentrations for tog11 and tog21
    fig3d.line(ts - tf_pulse[0], xs[:, circuit_name2pos['p_tog11']], line_width=2, line_color=bkRGB(100, 149, 237),
               legend_label='tog11')
    fig3d.line(ts - tf_pulse[0], xs[:, circuit_name2pos['p_tog21']], line_width=2, line_color=bkRGB(207, 181, 59),
               legend_label='tog21')

    # plot the toggle switch protein concentrations for tog12 and tog22
    fig3d.line(ts - tf_pulse[0], xs[:, circuit_name2pos['p_tog12']], line_width=2, line_color=bkRGB(127, 255, 212),
               legend_label='tog12')
    fig3d.line(ts - tf_pulse[0], xs[:, circuit_name2pos['p_tog22']], line_width=2, line_color=bkRGB(252, 194, 0),
               legend_label='tog22')

    # legend formatting
    fig3d.legend.location = "bottom_right"
    fig3d.legend.label_text_font_size = "8pt"

    # set fonts
    fig3d.xaxis.axis_label_text_font_size = "8pt"
    fig3d.xaxis.major_label_text_font_size = "8pt"
    fig3d.yaxis.axis_label_text_font_size = "8pt"
    fig3d.yaxis.major_label_text_font_size = "8pt"
    fig3d.yaxis[0].formatter = bkmodels.PrintfTickFormatter(format="%.0e")

    # save the figure
    bkplot.output_file(filename="fig3d.html", title="Figure 3 d")  # set up bokeh output file
    bkplot.save(fig3d)

    # PLOT FIGURE 3C - INTEGRASE CONCENTRATIONS AND GROWTH RATE
    # get the growth rate for plotting
    _, ls, _, _, _, _, _, _ = cellmodel_auxil.get_e_l_Fr_nu_psi_T_D_Dnohet(ts, xs, par, circuit_genes, circuit_miscs,
                                                                           circuit_name2pos)

    # initialise
    fig3e = bkplot.figure(
        frame_width=240,
        frame_height=180,
        x_axis_label="time since pulse start, h",
        y_axis_label="Cell growth rate, 1/h",
        x_range=(-5, tf_afterpulse[1] - tf_pulse[0]),
        y_range=(0.3, 1.3),
        tools="box_zoom,pan,hover,reset,save"
    )
    # set svg backend
    fig3e.output_backend = "svg"

    # add a line to show when synbthetic gene expression loss occurs
    fig3e.add_layout(
        bkmodels.PolyAnnotation(xs=[0, 0, tf_pulse[1] - tf_pulse[0], tf_pulse[1] - tf_pulse[0]],
                                ys=[0, 2, 2, 0],
                                line_width=0, line_alpha=0,
                                fill_color=bkRGB(100, 100, 100, 0.25)))
    fig3e.add_layout(bkmodels.Label(x=0, y=1.3,
                                    x_offset=0, y_offset=-145,
                                    text='Toggle 12 & 22 inducer pulse',
                                    text_font_size='8pt',
                                    angle=pi / 2))

    # plot the growth rate
    fig3e.line(ts - tf_pulse[0], np.array(ls), line_width=2, line_color=bkRGB(0, 0, 0), legend_label='Growth rate')

    # create an extra  y range for plotting integrase protein concentrations
    fig3e.extra_y_ranges = {"p_int": bkmodels.Range1d(start=0, end=250)}
    fig3e.add_layout(bkmodels.LinearAxis(y_range_name="p_int",
                                         axis_label="Integrase conc., nM",
                                         axis_line_color=bkRGB(255, 103, 0),
                                         major_tick_line_color=bkRGB(255, 103, 0),
                                         minor_tick_line_color=bkRGB(255, 103, 0)),
                     'right')  # add the alternative axis label to the figure

    # plot the integrase protein concentrations
    fig3e.line(ts - tf_pulse[0], xs[:, circuit_name2pos['p_int']], line_width=2, line_color=bkRGB(255, 103, 0),
               y_range_name="p_int", legend_label="Int. conc.")

    # set fonts
    fig3e.xaxis.axis_label_text_font_size = "8pt"
    fig3e.xaxis.major_label_text_font_size = "8pt"
    fig3e.yaxis.axis_label_text_font_size = "8pt"
    fig3e.yaxis.major_label_text_font_size = "8pt"

    # legend formatting
    fig3e.legend.location = "right"
    fig3e.legend.label_text_font_size = "8pt"

    # save the figure
    bkplot.output_file(filename="fig3e.html", title="Figure 3 e")  # set up bokeh output file
    bkplot.save(fig3e)

    return


# MAIN FUNCTION (FOR CALLING THE FIGURE GENERATION FUNCTIONS) ----------------------------------------------------------
def main():
    # set up jax
    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update("jax_enable_x64", True)

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

    # TOGGLES+PUNISHER: SIMULATION PARAMETERS --------------------------------------------------------------------------
    savetimestep = 0.1  # save time step
    rtol = 1e-6  # relative tolerance for the ODE solver
    atol = 1e-6  # absolute tolerance for the ODE solver

    # TOGGLES+PUNISHER: SIMULATING THE LOSS OF ONE TOGGLE SWITCH -------------------------------------------------------
    par_for_loseonetoggle = par.copy()
    lose_one_toggle('pun',  # short identifier of the case the output file names
                    par_for_loseonetoggle,  # dictionary with model parameters
                    ode_with_circuit,  # ODE function for the cell with synthetic circuit
                    circuit_F_calc,  # function for calculating the circuit's reaction rates
                    init_conds,  # initial condition DICTIONARY
                    circuit_genes, circuit_miscs,  # dictionaries with circuit gene and miscellaneous specie names
                    circuit_name2pos,  # species name to vector position decoder for the circuit
                    circuit_styles,  # dictionary with circuit gene and miscellaneous specie plotting styles
                    savetimestep, rtol, atol  # time step for saving the simulation, relative and absolute tolerances
                    )

    # # TOGGLES+PUNISHER: SIMULATING THE FLIPPING OF TOGGLE SWITCHES -----------------------------------------------------
    par_for_fliptoggles = par.copy()
    flip_toggles('pun',  # short identifier of the case the output file names
                 par_for_fliptoggles,  # dictionary with model parameters
                 ode_with_circuit,  # ODE function for the cell with synthetic circuit
                 circuit_F_calc,  # function for calculating the circuit's reaction rates
                 init_conds,  # initial condition DICTIONARY
                 circuit_genes, circuit_miscs,  # dictionaries with circuit gene and miscellaneous specie names
                 circuit_name2pos,  # species name to vector position decoder for the circuit
                 circuit_styles,  # dictionary with circuit gene and miscellaneous specie plotting styles
                 savetimestep, rtol, atol  # time step for saving the simulation, relative and absolute tolerances
                 )

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

    # ADDICTIVE TOGGLES: SIMULATING THE LOSS OF ONE TOGGLE SWITCH -------------------------------------------------------
    add_par_for_loseonetoggle = add_par.copy()
    # lose_one_toggle('add', # short identifier of the case the output file names
    #                 add_par_for_loseonetoggle,  # dictionary with model parameters
    #                 add_ode_with_circuit,  # ODE function for the cell with synthetic circuit
    #                 add_circuit_F_calc,  # function for calculating the circuit's reaction rates
    #                 add_init_conds,  # initial condition DICTIONARY
    #                 add_circuit_genes, add_circuit_miscs,  # dictionaries with circuit gene and miscellaneous specie names
    #                 add_circuit_name2pos,  # species name to vector position decoder for the circuit
    #                 add_circuit_styles,  # dictionary with circuit gene and miscellaneous specie plotting styles
    #                 savetimestep, rtol, atol  # time step for saving the simulation, relative and absolute tolerances
    #                 )
    return

# MAIN CALL ------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()