'''
FIG2_2A.PY - Make the plots illustrating how bifurcation determines the punisher's switching
'''
# By Kirill Sechkar

# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------
import jaxopt
import scipy
import pickle
from bokeh import plotting as bkplot, models as bkmodels, layouts as bklayouts, io as bkio
from bokeh import palettes as bkpalettes, transform as bktransform
from bokeh.colors import RGB as bkRGB
from math import pi

# OWN CODE IMPORTS -----------------------------------------------------------------------------------------------------
import sys
import os.path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from cell_model import *
from values_for_analytical import val4an
from switching_behaviour import *

# MAIN FUNCTION (FOR MAKING THE PLOTS) ---------------------------------------------------------------------------------
def main():
    # set up jax
    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update("jax_enable_x64", True)

    # CIRCUIT SPECIFICATION --------------------------------------------------------------------------------------------
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
    par['a_xtra'] = 2000.0

    par['c_switch'] = 10.0  # gene concentration (nM)
    par['a_switch'] = 100.0  # promoter strength (unitless)
    par['c_int'] = 10.0  # gene concentration (nM)
    par['a_int'] = 60.0  # promoter strength (unitless)
    par['d_int'] = 6.0  # integrase protein degradation rate (to avoid unnecessary punishment)
    par['c_cat'] = 10.0  # gene concentration (nM)
    par['a_cat'] = 500.0  # promoter strength (unitless)

    # punisher's transcription regulation function
    par['K_switch'] = 500.0  # 350.0  # Half-saturation constant for the self-activating switch gene promoter (nM)
    par['eta_switch'] = 3  # 2

    # Hill coefficient for the self-activating switch gene promoter (unitless)
    par['baseline_switch'] = 0.15  # 0.01  # Baseline value of the switch gene's transcription activation function
    par['p_switch_ac_frac'] = 0.45  # 0.75  # active fraction of protein (i.e. share of molecules bound by the inducer)

    # culture medium
    init_conds['s'] = 0.3
    par['h_ext'] = 10.5 * (10.0 ** 3)

    # get the values of the variables assumed to be constant
    simulate_to_find_cellvars = True  # get these values by simulation (True) or load from a saved file (False)
    if (simulate_to_find_cellvars):
        # get the cellular variables
        e, F_r, h, xi_a, xi_r, xi_cat, xi_switch_max, x_int_max, xi_other_genes = val4an(par, ode_with_circuit,
                                                                                         init_conds,
                                                                                         circuit_genes, circuit_miscs,
                                                                                         circuit_name2pos,
                                                                                         circuit_F_calc)
        # record the cellular variables
        cellvars = {'e': e, 'F_r': F_r,
                    'h': h,
                    'xi_a': xi_a, 'xi_r': xi_r, 'xi_other_genes': xi_other_genes, 'xi_cat': xi_cat,
                    'xi_switch_max': xi_switch_max, 'xi_int_max': x_int_max}
        pickle.dump(cellvars, open("cellvars.pkl", "wb"))
    else:
        cellvars = pickle.load(open("cellvars.pkl", "rb"))

    # FIND XI VALUES WITH ALL GENES AND WITH ONLY NATIVE AND CMR GENES -------------------------------------------------
    xi_native_cat = cellvars['xi_a'] + cellvars['xi_r'] + cellvars['xi_cat']
    xi_all = xi_native_cat + cellvars['xi_other_genes']

    # FIND THE THRESHOLD BIFURCATION -----------------------------------------------------------------------------------
    threshold_gfchanges_vector=np.array(threshold_gfchnages(par, cellvars))

    p_switch_threshold = threshold_gfchanges_vector[0]
    xi_threshold = threshold_gfchanges_vector[1]
    gfchange_int = threshold_gfchanges_vector[2]
    gfchange_F = threshold_gfchanges_vector[3]
    gfchange_intact = threshold_gfchanges_vector[4]

    # FIND REQUIRED AND REAL F_SWITCH VALUES FOR DIFFERENT VALUES OF XI ------------------------------------------------
    # on the x axis, switch protein concentrations
    p_switch_sup = pswitch_upper_bound_4nonsaddle(xi_threshold, par,cellvars)
    p_switch_axis=np.linspace(0,1.2*p_switch_sup,10000)

    # find real F_switch values
    F_reals = F_real_calc(p_switch_axis, par)

    # find required F_switch values
    F_reqs_all = F_req_calc(p_switch_axis, xi_all,par,cellvars)  # all synthetic genes present
    F_reqs_threshold = F_req_calc(p_switch_axis, xi_threshold,par,cellvars)  # threshold burden
    F_reqs_native_cat = F_req_calc(p_switch_axis, xi_native_cat,par,cellvars)  # only native genes and CAT

    # FIND FIXED POINTS FOR ALL BURDEN VALUES --------------------------------------------------------------------------
    # find for all synthetic genes present and just native and cat genes present
    findfps_func= lambda p_switch,xi: (F_real_calc(p_switch,par)-F_req_calc(p_switch,xi,par,cellvars))**2
    # just do it approximately by detecting minima in suqared difference of real and requiredF_switch values
    # for all genes
    fps_all = []
    findfpfunc_before = findfps_func(p_switch_axis[0],xi_all)
    findfpfunc = findfps_func(p_switch_axis[1],xi_all)
    for i in range(1,len(p_switch_axis)-1):
        findfpfunc_after = findfps_func(p_switch_axis[i+1],xi_all)
        if(findfpfunc_before>findfpfunc and findfpfunc_after>findfpfunc):
            fps_all.append([p_switch_axis[i],F_reals[i]])
        findfpfunc_before = findfpfunc
        findfpfunc = findfpfunc_after
    # for threshold burden
    fps_threshold = []
    findfpfunc_before = findfps_func(p_switch_axis[0],xi_threshold)
    findfpfunc = findfps_func(p_switch_axis[1],xi_threshold)
    for i in range(1,len(p_switch_axis)-1):
        findfpfunc_after = findfps_func(p_switch_axis[i+1],xi_threshold)
        if(findfpfunc_before>findfpfunc and findfpfunc_after>findfpfunc):
            fps_threshold.append([p_switch_axis[i],F_reals[i]])
        findfpfunc_before = findfpfunc
        findfpfunc = findfpfunc_after
    # for just native and cat genes
    fps_native_cat = []
    findfpfunc_before = findfps_func(p_switch_axis[0],xi_native_cat)
    findfpfunc = findfps_func(p_switch_axis[1],xi_native_cat)
    for i in range(1,len(p_switch_axis)-1):
        findfpfunc_after = findfps_func(p_switch_axis[i+1],xi_native_cat)
        if(findfpfunc_before>findfpfunc and findfpfunc_after>findfpfunc):
            fps_native_cat.append([p_switch_axis[i],F_reals[i]])
        findfpfunc_before = findfpfunc
        findfpfunc = findfpfunc_after
    fps_native_cat=fps_native_cat[1:]  # remove the first one, which is NOT a fixed point

    # convert to numpy arrays
    fps_all = np.array(fps_all)
    fps_threshold = np.array(fps_threshold)
    fps_native_cat = np.array(fps_native_cat)


    # PLOT -------------------------------------------------------------------------------------------------------------
    all_figure = bkplot.figure(
        frame_width=240,
        frame_height=180,
        x_axis_label="Switch protein conc.",
        y_axis_label='Fs value',
        x_range=(0, max(p_switch_axis)),
        y_range=(0, 1),
        title="Burden with synth. gene exp.",
        tools='pan,box_zoom,reset,save'
    )
    threshold_figure = bkplot.figure(
        frame_width=240,
        frame_height=180,
        x_axis_label="Switch protein conc.",
        y_axis_label='Fs value',
        x_range=(0, max(p_switch_axis)),
        y_range=(0, 1),
        title="Threshold gene exp. burden",
        tools='pan,box_zoom,reset,save'
    )
    native_cat_figure = bkplot.figure(
        frame_width=240,
        frame_height=180,
        x_axis_label="Switch protein conc.",
        y_axis_label='Fs value',
        x_range=(0, max(p_switch_axis)),
        y_range=(0, 1),
        title="Native & CAT gene exp. burden only",
        tools='pan,box_zoom,reset,save'
    )
    figures=[all_figure,threshold_figure,native_cat_figure]
    F_reqs=[F_reqs_all,F_reqs_threshold,F_reqs_native_cat]
    F_req_dashes=['dashed','dashdot','solid']
    fps=[fps_all,fps_threshold,fps_native_cat]
    for i in range(0,3):
        # svg backend
        figures[i].output_backend = "svg"

        # plot F values
        figures[i].line(p_switch_axis, F_reals, line_width=2, color=bkRGB(72, 209, 204), legend_label='Real')
        figures[i].line(p_switch_axis, F_reqs[i], line_width=2, color=bkRGB(0,0,0), line_dash=F_req_dashes[i],
                        legend_label='Required')

        # legend settings
        figures[i].legend.location = "top_left"
        figures[i].legend.label_text_font_size = '8pt'

        # handle axis ticks
        figures[i].yaxis.ticker = bkmodels.FixedTicker(ticks=[par['baseline_switch'],1])
        figures[i].yaxis.major_label_overrides = {par['baseline_switch']:'Fsb',1:'1'}
        # figures[i].xaxis.major_tick_line_color = None  # turn off x-axis major ticks
        # figures[i].xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
        # figures[i].yaxis.major_tick_line_color = None  # turn off y-axis major ticks
        # figures[i].yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
        figures[i].xaxis.major_label_text_font_size = '8pt'  # turn off x-axis tick labels
        figures[i].yaxis.major_label_text_font_size = '8pt'  # turn off y-axis tick labels
        figures[i].xaxis.ticker = bkmodels.FixedTicker(ticks=[0]+fps[i][:,0])
        figures[i].xaxis.major_label_overrides = {fps[i][j, 0]: '' for j in range(0, len(fps[i]))}

        # axis label and title font size
        figures[i].xaxis.axis_label_text_font_size = '8pt'
        figures[i].yaxis.axis_label_text_font_size = '8pt'
        figures[i].title.text_font_size = '8pt'

    # indicate the guaranteed fold change
    threshold_figure.line([fps_threshold[0, 0], fps_threshold[1, 0]], [fps_threshold[0, 1], fps_threshold[0, 1]],
                          line_width=1, line_color=bkRGB(0, 0, 0))
    threshold_figure.add_layout(bkmodels.Arrow(end=bkmodels.NormalHead(fill_color=bkRGB(0, 0, 0),
                                                                       line_color=bkRGB(0, 0, 0),
                                                                       size=5),
                                               line_color=bkRGB(0, 0, 0), line_width=1,
                                               x_start=fps_threshold[1, 0], x_end=fps_threshold[1, 0],
                                               y_start=fps_threshold[0, 1], y_end=fps_threshold[1, 1] - 0.025))
    threshold_figure.add_layout(bkmodels.Arrow(end=bkmodels.NormalHead(fill_color=bkRGB(0, 0, 0),
                                                                       line_color=bkRGB(0, 0, 0),
                                                                       size=5),
                                               line_color=bkRGB(0, 0, 0), line_width=1,
                                               x_start=fps_threshold[1, 0], x_end=fps_threshold[1, 0],
                                               y_start=fps_threshold[1, 1] - 0.025, y_end=fps_threshold[0, 1]))

    # mark the fixed points for all synthetic genes present
    all_figure.circle(fps_all[0, 0], fps_all[0, 1],
                      size=8, line_width=2, line_color=bkRGB(222, 49, 99), fill_color=bkRGB(222, 49, 99))
    all_figure.circle(fps_all[1, 0], fps_all[1, 1],
                      size=8, line_width=2, line_color=bkRGB(222, 49, 99), fill_color=bkRGB(222, 49, 99, 0))
    all_figure.circle(fps_all[2, 0], fps_all[2, 1],
                      size=8, line_width=2, line_color=bkRGB(222, 49, 99), fill_color=bkRGB(222, 49, 99))
    # add arrows indicating the stability of points for all synthetic genes present
    arrow_length=150
    arrows_start = 0
    arrows_end = fps_all[0, 0]
    arrows_range = np.arange(arrows_start, arrows_end+arrow_length, arrow_length)
    arrows_range[-1] = arrows_end
    for i in range(0, len(arrows_range)-1):
        all_figure.add_layout(bkmodels.Arrow(end=bkmodels.NormalHead(fill_color=bkRGB(200,200,200),
                                                                     line_color=bkRGB(200,200,200),
                                                                     size=10),
                                             line_color=bkRGB(200,200,200), line_width=2,
                                             x_start=arrows_range[i], x_end=arrows_range[i+1],
                                             y_start=0.05, y_end=0.05))
    arrows_start = fps_all[0, 0]
    arrows_end = fps_all[1, 0]
    arrows_range = np.arange(arrows_start, arrows_end+arrow_length, arrow_length)
    arrows_range[-1] = arrows_end
    for i in range(0, len(arrows_range) - 1):
        all_figure.add_layout(bkmodels.Arrow(end=bkmodels.NormalHead(fill_color=bkRGB(200,200,200),
                                                                     line_color=bkRGB(200,200,200),
                                                                     size=10),
                                             line_color=bkRGB(200,200,200), line_width=2,
                                             x_start=arrows_range[i+1], x_end=arrows_range[i],
                                             y_start=0.05, y_end=0.05))
    arrows_start = fps_all[1, 0]
    arrows_end = fps_all[2, 0]
    arrows_range = np.arange(arrows_start, arrows_end+arrow_length, arrow_length)
    arrows_range[-1] = arrows_end
    for i in range(0, len(arrows_range) - 1):
        all_figure.add_layout(bkmodels.Arrow(end=bkmodels.NormalHead(fill_color=bkRGB(200,200,200),
                                                                     line_color=bkRGB(200,200,200),
                                                                     size=10),
                                             line_color=bkRGB(200,200,200), line_width=2,
                                             x_start=arrows_range[i], x_end=arrows_range[i+1],
                                             y_start=0.05, y_end=0.05))
    arrows_start = fps_all[2, 0]
    arrows_end = p_switch_sup*1.2
    arrows_range = np.arange(arrows_start, arrows_end+arrow_length, arrow_length)
    arrows_range[-1] = arrows_end
    for i in range(0, len(arrows_range) - 1):
        all_figure.add_layout(bkmodels.Arrow(end=bkmodels.NormalHead(fill_color=bkRGB(200,200,200),
                                                                     line_color=bkRGB(200,200,200),
                                                                     size=10),
                                             line_color=bkRGB(200,200,200), line_width=2,
                                             x_start=arrows_range[i + 1], x_end=arrows_range[i],
                                             y_start=0.05, y_end=0.05))

    # mark the fixed points for threshold burden
    threshold_figure.circle(fps_threshold[0, 0], fps_threshold[0, 1],
                            size=8, line_width=2, line_color=bkRGB(222, 49, 99), fill_color=bkRGB(222, 49, 99, 0))
    threshold_figure.circle(fps_threshold[1, 0], fps_threshold[1, 1],
                            size=8, line_width=2, line_color=bkRGB(222, 49, 99), fill_color=bkRGB(222, 49, 99))
    # add arrows indicating the stability of points for threshold burden
    arrows_start=0
    arrows_end=fps_threshold[0, 0]
    arrows_range=np.arange(arrows_start, arrows_end+arrow_length, arrow_length)
    arrows_range[-1] = arrows_end
    for i in range(0, len(arrows_range)-1):
        threshold_figure.add_layout(bkmodels.Arrow(end=bkmodels.NormalHead(fill_color=bkRGB(200,200,200),
                                                                           line_color=bkRGB(200,200,200),
                                                                           size=10),
                                                   line_color=bkRGB(200,200,200), line_width=2,
                                                   x_start=arrows_range[i], x_end=arrows_range[i+1],
                                                   y_start=0.05, y_end=0.05))
    arrows_start = fps_threshold[0, 0]
    arrows_end = fps_threshold[1,0]
    arrows_range = np.arange(arrows_start, arrows_end+arrow_length, arrow_length)
    arrows_range[-1] = arrows_end
    for i in range(0, len(arrows_range) - 1):
        threshold_figure.add_layout(bkmodels.Arrow(end=bkmodels.NormalHead(fill_color=bkRGB(200,200,200),
                                                                           line_color=bkRGB(200,200,200),
                                                                           size=10),
                                                   line_color=bkRGB(200,200,200), line_width=2,
                                                   x_start=arrows_range[i], x_end=arrows_range[i+1],
                                                   y_start=0.05, y_end=0.05))
    arrows_start = fps_threshold[1,0]
    arrows_end = p_switch_sup*1.2
    arrows_range = np.arange(arrows_start, arrows_end+arrow_length, arrow_length)
    arrows_range[-1] = arrows_end
    for i in range(0, len(arrows_range) - 1):
        threshold_figure.add_layout(bkmodels.Arrow(end=bkmodels.NormalHead(fill_color=bkRGB(200,200,200),
                                                                           line_color=bkRGB(200,200,200),
                                                                           size=10),
                                                   line_color=bkRGB(200,200,200), line_width=2,
                                                   x_start=arrows_range[i + 1], x_end=arrows_range[i],
                                                   y_start=0.05, y_end=0.05))
    # mark the fixed points for native and cat gene expression burden
    native_cat_figure.circle(fps_native_cat[0, 0], fps_native_cat[0, 1],
                                size=8, line_width=2, line_color=bkRGB(222, 49, 99), fill_color=bkRGB(222, 49, 99))
    # add arrows indicating the stability of points for native and cat gene expression burden
    arrows_start = 0
    arrows_end = fps_native_cat[0, 0]
    arrows_range = np.arange(arrows_start,arrows_end+arrow_length,arrow_length)
    arrows_range[-1] = arrows_end
    for i in range(0,len(arrows_range)-1):
        native_cat_figure.add_layout(bkmodels.Arrow(end=bkmodels.NormalHead(fill_color=bkRGB(200,200,200),
                                                                              line_color=bkRGB(200,200,200),
                                                                                size=10),
                                                        line_color=bkRGB(200,200,200), line_width=2,
                                                        x_start=arrows_range[i], x_end=arrows_range[i+1],
                                                        y_start=0.05, y_end=0.05))
    arrows_start = fps_native_cat[0, 0]
    arrows_end = p_switch_sup*1.2
    arrows_range = np.arange(arrows_start, arrows_end+arrow_length, arrow_length)
    arrows_range[-1] = arrows_end
    for i in range(0, len(arrows_range) - 1):
        native_cat_figure.add_layout(bkmodels.Arrow(end=bkmodels.NormalHead(fill_color=bkRGB(200,200,200),
                                                                            line_color=bkRGB(200,200,200),
                                                                            size=10),
                                                    line_color=bkRGB(200,200,200), line_width=2,
                                                    x_start=arrows_range[i+1], x_end=arrows_range[i],
                                                    y_start=0.05, y_end=0.05))

    # label the guaranteed fold-change
    threshold_figure.add_layout(bkmodels.Label(x=fps_threshold[1, 0], y=0.5,
                                               x_offset=2, y_offset=0,
                                               text='Guaranteed fold-change \n in Fswitch', text_font_size='8pt',
                                               text_align='center', text_baseline='middle',
                                               angle=-pi / 2
                                               ))

    # save plots
    bkplot.output_file('bifurcation_illustration.html', title='Bifurcation illustration')
    bkplot.save(bklayouts.grid([all_figure,threshold_figure,native_cat_figure]))
    return

# MAIN CALL ------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()