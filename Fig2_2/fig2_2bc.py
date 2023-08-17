'''
FIG2_2BC.PY - Study the switching behaviour of the punisher circuit.
'''
# By Kirill Sechkar

# PACKAGE IMPORTS ------------------------------------------------------------------------------------------------------
import jaxopt
import pickle
from bokeh import palettes as bkpalettes, transform as bktransform
from bokeh.colors import RGB as bkRGB
from contourpy import contour_generator as cgen

# OWN CODE IMPORTS -----------------------------------------------------------------------------------------------------
import sys
import os.path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from cell_model import *
from values_for_analytical import val4an


# FINDING THE THRESHOLD BIFURCATION: PARAMETRIC APPROACH ---------------------------------------------------------------
# difference of gradients at the fixed point for a given value of F_switch
def gradiff_from_F(F, par, cellvars):
    # reconstruct p_switch and xi
    p_switch = pswitch_from_F(F, par)
    xi = xi_from_F_and_pswitch(F, p_switch, par, cellvars)

    # get the gradients
    dFreal_dpswitch = dFreal_dpswitch_calc(p_switch, par)
    dFreq_dpswitch = dFreq_dpswitch_calc(p_switch, xi, par, cellvars)
    return dFreal_dpswitch - dFreq_dpswitch


# difference of gradients at the fixed point for a given value of p_switch
def gradiff_from_pswitch(p_switch, par, cellvars):
    # reconstruct F_switch and xi
    F = F_real_calc(p_switch, par)
    xi = xi_from_F_and_pswitch(F, p_switch, par, cellvars)

    # get the gradients
    dFreal_dpswitch = dFreal_dpswitch_calc(p_switch, par)
    dFreq_dpswitch = dFreq_dpswitch_calc(p_switch, xi, par, cellvars)
    return dFreal_dpswitch - dFreq_dpswitch


# find p_switch value for a given real value of F_switch
def pswitch_from_F(F, par):
    return (par['K_switch'] / par['p_switch_ac_frac']) * \
        ((F - par['baseline_switch']) / (1 - F)) ** (1 / par['eta_switch'])


# find the value of xi yielding a fixed point for given F_switch and p_switch values
def xi_from_F_and_pswitch(F, p_switch, par, cellvars):
    return F / p_switch * (cellvars['xi_switch_max'] + cellvars['xi_int_max']) * \
        (par['M'] * (1 - par['phi_q']) / par['n_switch'] * cellvars['xi_switch_max'] / (
                cellvars['xi_switch_max'] + cellvars['xi_int_max']) - p_switch)


# just for convenience, can get a pair of xi and p_switch values
def pswitch_and_xi_from_F(F, par, cellvars):
    p_switch = pswitch_from_F(F, par)
    xi = xi_from_F_and_pswitch(F, p_switch, par, cellvars)
    return jnp.array([p_switch,xi])


# upper bound for xi to find the saddle point at the bifurcation (inflexion in real F_switch values)
def pswitch_inflexion_in_Freal(par):
    return ((par['eta_switch'] - 1) / (par['eta_switch'] + 1)) ** (1 / par['eta_switch']) * (
                par['K_switch'] / par['p_switch_ac_frac'])


# FINDING THE GUARANTEED CHANGE IN P_INT EXPRESSION --------------------------------------------------------------------
# find the difference of values between the real and required F_switch values
# (used for optimisation in terms of p_switch)
def validff_to_find_pswitch(p_switch,
                            xi,
                            par, cellvars):
    return F_real_calc(p_switch, par) - F_req_calc(p_switch, xi, par, cellvars)


# upper bound for p_switch to find the non-saddle fixed point - for a given (threshold) value of xi
def pswitch_upper_bound_4nonsaddle(xi,
                                   par, cellvars):
    return cellvars['xi_switch_max'] / (cellvars['xi_switch_max'] + cellvars['xi_int_max'] +
                                        xi) * par['M'] * (1 - par['phi_q']) / par['n_switch']

# find the integrase protein concentration for a given value of p_switch and burden xi
def pint_from_pswitch_and_xi(p_switch,
                             xi,
                             par, cellvars):
    # get the F value
    F = F_real_calc(p_switch, par)

    # get the chloramphenicol-dependent factor
    cm_factor = (par['K_D']+cellvars['h'])/par['K_D']

    # get the ribosome concentration
    R = par['M'] / ((1 - par['phi_q']) * par['n_r']) * \
        cellvars['xi_r'] / (F * (cellvars['xi_switch_max'] + cellvars['xi_int_max']) + xi)

    # get the growth rate estimate, assuming a negligible share of idle ribosomes
    l = cellvars['e'] * R / par['M']

    # get m_i/k_i values from burden values for all genes
    maka=cellvars['xi_a']* l / (par['b_a']+l)
    mrkr = cellvars['xi_r'] * l / (par['b_r'] + l)
    motherskothers = cellvars['xi_other_genes'] * l / (par['b_switch'] + l)  # given that our derivations hinge on mRNA degradation rates being similar and this is a collective entry for many synthetic genes, we just plug in b_switch
    mcatkcat = cellvars['xi_cat'] * l / (par['b_cat'] + l)
    mswitchkswitch = F * cellvars['xi_switch_max'] * l / (par['b_switch'] + l)
    mintkint = F * cellvars['xi_int_max'] * l / (par['b_int'] + l)

    # sum the m_i/k_i value
    mk_sum= maka + mrkr + motherskothers + mcatkcat + mswitchkswitch + mintkint

    # numerator: steady-state production rate of p_int; denominator: steady-state degradation rate of p_int (per nM)
    return (cellvars['e'] * R / par['n_int']) * mintkint / (cm_factor * (1+mk_sum/(1-par['phi_q']))) / (l + par['d_int'])


# F_SWITCH FUNCTION VALUES ---------------------------------------------------------------------------------------------
# real value
def F_real_calc(p_switch, par):
    K_div_ac_frac = par['K_switch'] / par['p_switch_ac_frac']
    p_switch_dependent_term = (p_switch * par['p_switch_ac_frac'] / par['K_switch']) ** par['eta_switch']
    return par['baseline_switch'] + (1 - par['baseline_switch']) * (
            p_switch_dependent_term / (p_switch_dependent_term + 1))


# required value
def F_req_calc(p_switch, xi, par, cellvars):
    return p_switch * xi / (cellvars['xi_switch_max'] + cellvars['xi_int_max']) / \
        (par['M'] * (1 - par['phi_q']) / par['n_switch'] * cellvars['xi_switch_max'] / (
                cellvars['xi_switch_max'] + cellvars['xi_int_max']) - p_switch)


# F_SWITCH FUNCTION GRADIENTS ------------------------------------------------------------------------------------------
# real value
def dFreal_dpswitch_calc(p_switch, par):
    K_div_ac_frac = par['K_switch'] / par['p_switch_ac_frac']
    return par['eta_switch'] * (1 - par['baseline_switch']) * K_div_ac_frac ** par[
        'eta_switch'] * p_switch ** (par['eta_switch'] - 1) / \
        (K_div_ac_frac ** par['eta_switch'] + p_switch ** par['eta_switch']) ** 2


# required value
def dFreq_dpswitch_calc(p_switch, xi, par, cellvars):
    MQns = par['M'] * (1 - par['phi_q']) / par['n_switch']
    return xi * cellvars['xi_switch_max'] * MQns / \
        (p_switch * (cellvars['xi_switch_max'] + cellvars['xi_int_max']) - cellvars['xi_switch_max'] * MQns) ** 2


# AUXILIARY FINCTIONS FOR DIAGNOSTICS AND CASE INVESTIGATION -----------------------------------------------------------
# Return squared difference between the real and required F_switch values and their gradients
def differences_squared(pswitch_xi,# decision variables: p_switch and xi
                        par,  # dictionary with model parameters
                        cellvars  # cellular variables that we assume to be constant
                        ):
    # unpack the decision variables
    p_switch = pswitch_xi[0]
    xi = pswitch_xi[1]

    # find the values of F_switch functions and their gradients
    F_real = F_real_calc(p_switch, par)
    F_req = F_req_calc(p_switch, xi, par, cellvars)
    dFreal_dpswitch = dFreal_dpswitch_calc(p_switch, par)
    dFreq_dpswitch = dFreq_dpswitch_calc(p_switch, xi, par, cellvars)

    return jnp.array([(F_real - F_req) ** 2, (dFreal_dpswitch - dFreq_dpswitch) ** 2])

# CHECK IF THE THRESHOLD/BIFURCATION POINT EXISTS ----------------------------------------------------------------------
# to do so, it is enough to check the sign of the difference of gradients in the inflexion point of real F_switch values
def check_if_threshold_exists(par, cellvars):
    p_switch_inflexion = pswitch_inflexion_in_Freal(par) # get p_switch value in the inflexion point
    return gradiff_from_pswitch(p_switch_inflexion, par, cellvars) > 0 # TRUE <=> threshold exists


# FIND THRESHOLD AND GUARANTEED FOLD EXPRESSION CHANGES ----------------------------------------------------------------
# find the threshold and guaranteed fold expression changes for a given set of parameters
def threshold_gfchnages(par, cellvars):
    pswitch_inflexion = pswitch_inflexion_in_Freal(
        par)  # upper bound of feasible region for p_switch (inflexion point in real F_switch)
    F_upper_bound = F_real_calc(pswitch_inflexion, par)  # upper bound of feasible region for F
    F_lower_bound = par['baseline_switch']  # lower bound of feasible region for F (corresponds to p_switch=0)

    # FIND THE THRESHOLD BIFURCATION POINT -----------------------------------------------------------------------------
    # create an instance of the optimisation problem
    threshold_problem = jaxopt.Bisection(optimality_fun=gradiff_from_F,
                                         lower=F_lower_bound, upper=F_upper_bound,
                                         maxiter=10000, tol=1e-18,
                                         check_bracket=False)  # required for vmapping and jitting

    # solve the optimisation problem
    F_threshold = threshold_problem.run(par=par, cellvars=cellvars).params
    # unpack the solution
    p_switch_threshold=pswitch_from_F(F_threshold,par)
    xi_threshold = xi_from_F_and_pswitch(F_threshold,p_switch_threshold,par,cellvars)


    # FIND THE GUARANTEED CHANGE IN INTEGRASE PROTEIN EXPRESSION -------------------------------------------------------
    # find the non-saddle fixed point for the threshold xi value
    p_switch_sup = pswitch_upper_bound_4nonsaddle(xi_threshold, par,
                                                  cellvars)  # supremum of biologically possible p_switch values for a given xi
    nonsaddle_problem = jaxopt.Bisection(optimality_fun=validff_to_find_pswitch,
                                         lower=pswitch_inflexion, upper=p_switch_sup,  # non-saddle f.p. is found between the inflexion and the supremum
                                         maxiter=10000, tol=1e-18,
                                         check_bracket=False)  # required for vmapping and jitting

    p_switch_nonsaddle = nonsaddle_problem.run(xi=xi_threshold, par=par,
                                                   cellvars=cellvars).params  # optimise to find the p_switch value at non-saddle fixed point

    # find the integrase protein expression at the saddle and non-saddle fixed point
    p_int_saddle = pint_from_pswitch_and_xi(p_switch_threshold, xi_threshold, par, cellvars)
    p_int_nonsaddle = pint_from_pswitch_and_xi(p_switch_nonsaddle, xi_threshold, par, cellvars)

    # find the guaranteed fold-change in integrase protein expression
    gfchange_int = p_int_nonsaddle / p_int_saddle

    # find the guaranteed fold-change in F value (just as an extra)
    gfchange_F = F_real_calc(p_switch_nonsaddle, par) / F_threshold

    # find the guaranteed fold-change in integrase activity
    intact_saddle = (p_int_saddle/par['K_bI~'])**4/ (1+(p_int_saddle/par['K_bI~'])**4)
    intact_nonsaddle = (p_int_nonsaddle/par['K_bI~'])**4/ (1+(p_int_nonsaddle/par['K_bI~'])**4)
    gfchange_intact = intact_nonsaddle/intact_saddle

    return jnp.array([p_switch_threshold, xi_threshold, gfchange_int, gfchange_F, gfchange_intact])


# DETAILED PLOTS FOR A SINGLE CASE -------------------------------------------------------------------------------------
# pass negative first three arguments when there is no threshold vbifurcation point
def plots_for_single_case(p_switch_threshold,  # p_switch value at the threshold bifurcation point
                          xi_threshold,  # xi value yielding the threshold bifurcation point
                          gfchange_F,  # guaranteed fold-change in F at xi_threshold
                          par, cellvars):
    # find the supremum of biologically possible p_switch values for a given xi
    p_switch_sup = pswitch_upper_bound_4nonsaddle(xi_threshold, par,
                                                  cellvars)

    # find the non-saddle point
    p_switch_nonsaddle = pswitch_from_F(gfchange_F* F_real_calc(p_switch_threshold, par),  # F value at the non-saddle point
                                        par)  # model parameters


    # get a parameterised line giving the fixed points
    F_axis = jnp.linspace(0.01, F_real_calc(p_switch_sup, par), 100)
    pswitch_xi_fp = jax.jit(jax.vmap(pswitch_and_xi_from_F, in_axes=(0, None, None)))(F_axis, par, cellvars)

    # make a mesh
    xi_mesh_axis = jnp.linspace(0.1, 2 * (sum(cellvars.values()) - cellvars['e'] - cellvars['F_r']), 100)
    p_switch_mesh_axis = jnp.linspace(0.1, p_switch_sup, 100)
    p_switch_mesh, xi_mesh = jnp.meshgrid(p_switch_mesh_axis, xi_mesh_axis)
    p_switch_mesh_ravel = p_switch_mesh.ravel()
    xi_mesh_ravel = xi_mesh.ravel()
    stacked_mesh = jnp.stack((p_switch_mesh_ravel, xi_mesh_ravel), axis=1)
    # vmap the objective function
    diffs_squared_for_xipswitch = lambda pswitch_xi: differences_squared(pswitch_xi, par, cellvars)
    vmapped_opt4threshold_for_pswitchxi = jax.jit(jax.vmap(diffs_squared_for_xipswitch, in_axes=0))
    objfun_vals_array = vmapped_opt4threshold_for_pswitchxi(stacked_mesh)
    valdiff_vals = objfun_vals_array[:, 0]
    gradiff_vals = objfun_vals_array[:, 1]
    product_vals = gradiff_vals * valdiff_vals

    # find burden for native genes and cat only
    xi_native_cat = cellvars['xi_a'] + cellvars['xi_r'] + cellvars['xi_cat']
    # find burden with xtra gene expression
    xi_with_other_genes = xi_native_cat + cellvars['xi_other_genes']

    # plot the squaed difference of values
    bkplot.output_file('single_case.html', title='Looking at a single case in detail')
    valdiff_figure = bkplot.figure(title='Squared difference of values',
                                   x_axis_label='xi',
                                   y_axis_label='p_switch',
                                   frame_height=255,
                                   frame_width=340,
                                   x_range=(np.array(xi_mesh_axis)[0], np.array(xi_mesh_axis)[-1]),
                                   y_range=(np.array(p_switch_mesh_axis)[0], np.array(p_switch_mesh_axis)[-1]),
                                   tools='pan,box_zoom,hover,reset'
                                   )
    levels = np.logspace(np.log10(min(valdiff_vals)), np.log10(max(valdiff_vals)), 25)
    contour_renderer = valdiff_figure.contour(np.array(xi_mesh_axis), np.array(p_switch_mesh_axis),
                                              np.array(valdiff_vals).reshape(len(p_switch_mesh_axis),
                                                                             len(xi_mesh_axis)).T,
                                              # NOTE: transposed!!!
                                              line_color='black', fill_color=bkpalettes.Plasma256, levels=levels)
    colourbar = contour_renderer.construct_color_bar()
    valdiff_figure.add_layout(colourbar, "right")
    # add the fixed points line
    valdiff_figure.line(np.array(pswitch_xi_fp)[:, 1], np.array(pswitch_xi_fp)[:, 0],
                        line_color='white', line_width=2.5)
    if (p_switch_threshold > 0.0):
        # show the threshold bifurcation point
        valdiff_figure.circle(xi_threshold, p_switch_threshold, size=10, color='white')
        # show the corresponding non-saddle fixed point
        valdiff_figure.diamond(xi_threshold, p_switch_nonsaddle, size=10, color='white')
    # add the relevant burden bounds
    valdiff_figure.line((xi_native_cat, xi_native_cat), (np.array(p_switch_mesh_axis)[0], np.array(p_switch_mesh_axis)[-1]),
                        line_color='black', line_width=2.5)
    valdiff_figure.line((xi_with_other_genes, xi_with_other_genes), (np.array(p_switch_mesh_axis)[0], np.array(p_switch_mesh_axis)[-1]),
                        line_color='black', line_width=2.5, line_dash='dashed')

    # plot the squared difference of gradients
    gradiff_figure = bkplot.figure(title='Squared difference of gradients',
                                   x_axis_label='xi',
                                   y_axis_label='p_switch',
                                   frame_height=255,
                                   frame_width=340,
                                   x_range=(np.array(xi_mesh_axis)[0], np.array(xi_mesh_axis)[-1]),
                                   y_range=(np.array(p_switch_mesh_axis)[0], np.array(p_switch_mesh_axis)[-1]),
                                   tools='pan,box_zoom,hover,reset'
                                   )
    levels = np.logspace(np.log10(min(gradiff_vals)), np.log10(max(gradiff_vals)), 25)
    contour_renderer = gradiff_figure.contour(np.array(xi_mesh_axis), np.array(p_switch_mesh_axis),
                                              np.array(gradiff_vals).reshape(len(xi_mesh_axis),
                                                                             len(p_switch_mesh_axis)).T,
                                              # NOTE: transposed!!!
                                              line_color='black', fill_color=bkpalettes.Plasma256, levels=levels)
    colourbar = contour_renderer.construct_color_bar()
    gradiff_figure.add_layout(colourbar, "right")
    if (p_switch_threshold > 0.0):
        # show the threshold bifurcation point
        gradiff_figure.circle(xi_threshold, p_switch_threshold, size=10, color='white')
        # show the corresponding non-saddle fixed point
        gradiff_figure.diamond(xi_threshold, p_switch_nonsaddle, size=10, color='white')
    # add the relevant burden bounds
    gradiff_figure.line((xi_native_cat, xi_native_cat), (np.array(p_switch_mesh_axis)[0], np.array(p_switch_mesh_axis)[-1]),
                        line_color='black',line_width=2.5)
    gradiff_figure.line((xi_with_other_genes, xi_with_other_genes), (np.array(p_switch_mesh_axis)[0], np.array(p_switch_mesh_axis)[-1]),
                        line_color='black',line_width=2.5, line_dash='dashed')

    # plot the objective function values
    product_figure = bkplot.figure(title='Product of squared differences',
                                   x_axis_label='xi',
                                   y_axis_label='p_switch',
                                   frame_height=255,
                                   frame_width=340,
                                   x_range=(np.array(xi_mesh_axis)[0], np.array(xi_mesh_axis)[-1]),
                                   y_range=(np.array(p_switch_mesh_axis)[0], np.array(p_switch_mesh_axis)[-1]),
                                   tools='pan,box_zoom,hover,reset'
                                   )
    levels = np.logspace(np.log10(min(product_vals)), np.log10(max(product_vals)), 25)
    contour_renderer = product_figure.contour(np.array(xi_mesh_axis), np.array(p_switch_mesh_axis),
                                              np.array(product_vals).reshape(len(xi_mesh_axis),
                                                                             len(p_switch_mesh_axis)).T,
                                              # NOTE: transposed!!!
                                              line_color='black', fill_color=bkpalettes.Plasma256, levels=levels)
    colourbar = contour_renderer.construct_color_bar()
    product_figure.add_layout(colourbar, "right")
    if (p_switch_threshold > 0.0):
        # show the threshold bifurcation point
        product_figure.circle(xi_threshold, p_switch_threshold, size=10, color='white')
        # show the corresponding non-saddle fixed point
        product_figure.diamond(xi_threshold, p_switch_nonsaddle, size=10, color='white')
    # add the relevant burden bounds
    product_figure.line((xi_native_cat, xi_native_cat), (np.array(p_switch_mesh_axis)[0], np.array(p_switch_mesh_axis)[-1]),
                        line_color='black', line_width=2.5)
    product_figure.line((xi_with_other_genes, xi_with_other_genes), (np.array(p_switch_mesh_axis)[0], np.array(p_switch_mesh_axis)[-1]),
                        line_color='black', line_width=2.5, line_dash='dashed')

    # PLOT F_SWITCH VALUES (TO ENSURE THE CALCULATIONS ARE RIGHT) ------------------------------------------------------
    if (p_switch_threshold > 0):
        # maximum biologically possible value of p_switch
        p_switch_axis = np.linspace(1, p_switch_sup, 1000)
        F_reals = F_real_calc(p_switch_axis, par)
        F_reqs = F_req_calc(p_switch_axis, xi_threshold, par, cellvars)
        F_figure = bkplot.figure(
            frame_height=255,
            frame_width=340,
            x_axis_label="p_s",
            y_axis_label="F_s",
            y_range=(0, 1),
            #title="Threshold bifurcation: plot of F_switch values",
        )
        F_figure.line(p_switch_axis, F_reals, color="red", legend_label='real', line_width=2)
        F_figure.line(p_switch_axis, F_reqs, color="blue", legend_label='required', line_width=2)
        F_figure.circle(p_switch_threshold, F_real_calc(p_switch_threshold, par), color="green", size=10)
        F_figure.x(p_switch_nonsaddle, F_real_calc(p_switch_nonsaddle, par), color="green", line_width=3, size=10)
        F_figure.legend.location = "bottom_right"

        # add the relevant burden bounds
        F_reqs_native_cat = F_req_calc(p_switch_axis, xi_native_cat, par, cellvars)
        F_reqs_with_other_genes = F_req_calc(p_switch_axis, xi_with_other_genes, par, cellvars)
        F_figure.line(p_switch_axis, F_reqs_native_cat, color="black", line_width=2.5)
        F_figure.line(p_switch_axis, F_reqs_with_other_genes, color="black", line_width=2.5, line_dash='dashed')

    else:
        F_figure = None

    # SAVE PLOTS -------------------------------------------------------------------------------------------------------
    bkplot.save(bklayouts.grid([[valdiff_figure, gradiff_figure, product_figure],
                                [F_figure, None, None]]))
    return


# MAKE THE PLOT FOR ETA AND RECIPROCAL OF BASELINE ---------------------------------------------------------------------
def eta_reciprocalbaseline_plot(eta_range, reciprocal_baseline_range,  # parameter ranges to consider
                                par, cellvars,  # model parameters and cellular variable values
                                xi_contours=None  # details of non-default burden contours to draw on the heatmap, if any
                                ):
    baseline_range = 1 / reciprocal_baseline_range

    # get a mesh grid, then flatten its x and y coordinates into a single linear array
    eta_mesh, baseline_mesh = np.meshgrid(eta_range, baseline_range)
    eta_mesh_ravel = eta_mesh.ravel()
    baseline_mesh_ravel = baseline_mesh.ravel()

    # specify vmapping axes
    par_vmapping_axes = {}
    for key in par.keys():
        if (key == 'eta_switch' or key == 'baseline_switch'):
            par_vmapping_axes[key] = 0
        else:
            par_vmapping_axes[key] = None
    cellvars_vmapping_axes = {}
    for key in cellvars.keys():
        cellvars_vmapping_axes[key] = None

    # DETERMINE WHICH PARAMETER COMBINATIONS YIELD A DESIRED THRESHOLD BIFURCATION POINT -------------------------------
    # make a vmappable parameter dictionary
    par_for_existence = par.copy()
    par_for_existence['eta_switch'] = jnp.array(eta_mesh_ravel)
    par_for_existence['baseline_switch'] = jnp.array(baseline_mesh_ravel)

    # make the checking function vmappable
    vmapped_check_if_threshold_exists = jax.jit(jax.vmap(check_if_threshold_exists,
                                                         in_axes=(par_vmapping_axes, cellvars_vmapping_axes)))

    # run
    threshold_exists = np.array(vmapped_check_if_threshold_exists(par_for_existence, cellvars))

    # FOR THOSE COMBINATIONS WHERE T/B POINT EXISTS, FIND IT AND THE GUARANTEED FOLD CHANGES ---------------------------
    # only consider parameter combinations where the threshold bifurcation point exists
    indices_where_threshold_exists = []
    for i in range(0, len(threshold_exists)):
        if (threshold_exists[i]):
            indices_where_threshold_exists.append(i)
    eta_mesh_ravel_exists = eta_mesh_ravel[indices_where_threshold_exists]
    baseline_mesh_ravel_exists = baseline_mesh_ravel[indices_where_threshold_exists]

    # make a vmappable parameter dictionary
    par_for_threshold_gfchanges = par.copy()
    par_for_threshold_gfchanges['eta_switch'] = jnp.array(eta_mesh_ravel_exists)
    par_for_threshold_gfchanges['baseline_switch'] = jnp.array(baseline_mesh_ravel_exists)

    # make the threshold and guaranteed fold change retrieval function vmappable
    vmapped_threshold_gfchnages = jax.jit(jax.vmap(threshold_gfchnages,
                                                   in_axes=(par_vmapping_axes, cellvars_vmapping_axes)))

    # run
    thresholds_gfchanges = np.array(vmapped_threshold_gfchnages(par_for_threshold_gfchanges, cellvars))

    # unpack the results
    p_switch_thresholds = thresholds_gfchanges[:, 0]
    xi_thresholds = thresholds_gfchanges[:, 1]
    gfchange_ints = thresholds_gfchanges[:, 2]
    gfchange_Fs = thresholds_gfchanges[:, 3]
    gfchange_intact = thresholds_gfchanges[:, 4]

    # FIND BURDEN CONTOURS  --------------------------------------------------------------------------------------------
    # fill the points where no threshold exists with INFS
    xi_thresholds_for_contour_ravel = np.zeros(threshold_exists.shape)  # initialise
    last_index_in_exist_list = 0
    for i in range(0, len(xi_thresholds_for_contour_ravel)):
        if (i == indices_where_threshold_exists[last_index_in_exist_list]):
            xi_thresholds_for_contour_ravel[i] = xi_thresholds[last_index_in_exist_list]
            if (last_index_in_exist_list < len(indices_where_threshold_exists) - 1):
                last_index_in_exist_list += 1
        else:
            xi_thresholds_for_contour_ravel[i] = np.inf
    xi_thresholds_for_contour = xi_thresholds_for_contour_ravel.reshape(len(eta_range),
                                                                        len(reciprocal_baseline_range)).T

    # create a contour generator
    threshold_cgen = cgen(x=reciprocal_baseline_range, y=eta_range,
                          z=xi_thresholds_for_contour)

    # if no xi values specified, plot one for all synthetic genes present and the one with just native genes and CAT
    if (xi_contours == None):
        xi_native_cat = cellvars['xi_a'] + cellvars['xi_r'] + cellvars['xi_cat']
        xi_with_all_genes = xi_native_cat + cellvars['xi_other_genes']
        xi_contours = {'values': [xi_native_cat, xi_with_all_genes],
                       'legends': ['Native & CAT \n gene exp. burden \n only', 'Burden \n w/ synth. \n gene exp.'],
                       'dashes': ['solid', 'dashed']}

    # find burden contour lines
    xi_contours['contour lines'] = []
    for i in range(0, len(xi_contours['values'])):
        xi_contours['contour lines'].append(threshold_cgen.lines(xi_contours['values'][i]))

    # PLOT THE RESULTS -------------------------------------------------------------------------------------------------
    # calculate widths and heights for heatmap rectangles - unintuitive due to log scale
    rect_widths_along_x_axis = np.zeros(len(reciprocal_baseline_range))
    rect_widths_along_x_axis[0] = reciprocal_baseline_range[1] - reciprocal_baseline_range[0]
    for i in range(1, len(reciprocal_baseline_range)):
        rect_widths_along_x_axis[i] = ((reciprocal_baseline_range[i] - reciprocal_baseline_range[i - 1]) -
                                       rect_widths_along_x_axis[i - 1] / 2) * 2
    rect_heights_along_y_axis = np.zeros(len(eta_range))
    rect_heights_along_y_axis[0] = eta_range[1] - eta_range[0]
    for i in range(1, len(eta_range)):
        rect_heights_along_y_axis[i] = ((eta_range[i] - eta_range[i - 1]) - rect_heights_along_y_axis[i - 1] / 2) * 2
    rect_widths_ravel_exists = np.zeros(baseline_mesh_ravel_exists.shape)
    rect_heights_ravel_exists = np.zeros(eta_mesh_ravel_exists.shape)
    for i in range(0, len(baseline_mesh_ravel_exists)):
        baseline_where = np.argwhere(
            baseline_range == baseline_mesh_ravel_exists[i])  # locate the baseline value in the baseline range
        rect_widths_ravel_exists[i] = rect_widths_along_x_axis[baseline_where[0][0]]*1.25
        eta_where = np.argwhere(eta_range == eta_mesh_ravel_exists[i])  # locate the eta value in the eta range
        rect_heights_ravel_exists[i] = rect_heights_along_y_axis[eta_where[0][0]]*1.25

    # make a dataframe for the heatmap of guaranteed fold changes
    heatmap_df = pd.DataFrame({'eta_s': eta_mesh_ravel_exists, 'reciprocal_baseline_s': 1 / baseline_mesh_ravel_exists,
                               'gfchange_F': gfchange_Fs, 'gfchange_int': gfchange_ints, 'gfchange_intact': gfchange_intact,
                               'rect_width': rect_widths_ravel_exists, 'rect_height': rect_heights_ravel_exists})

    # PLOT guaranteed fold-changes
    gfchange_F_figure = bkplot.figure(
        frame_width=240,
        frame_height=180,
        x_axis_label="Maximum-to-baseline expression ratio",
        y_axis_label="Hill coefficient",
        x_range=(min(reciprocal_baseline_range), max(reciprocal_baseline_range)),
        y_range=(min(eta_range), max(eta_range)),
        x_axis_type="log",
        #title="F_switch value GF-changes",
        tools='pan,box_zoom,reset,save'
    )
    gfchange_int_figure = bkplot.figure(
        frame_width=240,
        frame_height=180,
        x_axis_label="Maximum-to-baseline expression ratio",
        y_axis_label="Hill coefficient",
        x_range=(min(reciprocal_baseline_range), max(reciprocal_baseline_range)),
        y_range=(min(eta_range), max(eta_range)),
        x_axis_type="log",
        #title="Integrase expression GF-changes",
        tools='pan,box_zoom,reset,save'
    )
    gfchange_intact_figure = bkplot.figure(
        frame_width=240,
        frame_height=180,
        x_axis_label="Maximum-to-baseline expression ratio",
        y_axis_label="Hill coefficient",
        x_range=(min(reciprocal_baseline_range), max(reciprocal_baseline_range)),
        y_range=(min(eta_range), max(eta_range)),
        x_axis_type="log",
        #title="Integrase activity GF-changes",
        tools='pan,box_zoom,reset,save'
    )
    what_to_plot = ['gfchange_F', 'gfchange_int', 'gfchange_intact']
    which_ticks = [[1,4,16,64],[1,4,16,64],[1,10,100,1000,10000]]
    what_to_plot_cntr = 0
    for figure in [gfchange_F_figure, gfchange_int_figure, gfchange_intact_figure]:
        # svg backend
        figure.output_backend= "svg"
        # set x ticks
        figure.xaxis.ticker = bkmodels.FixedTicker(ticks=[1, 10, 100],minor_ticks=[2, 4, 6, 8, 20, 40, 60, 80])
        figure.xaxis.formatter = bkmodels.PrintfTickFormatter(format="%d")
        # plot the heatmap itself
        rect=figure.rect(x="reciprocal_baseline_s", y="eta_s", source=heatmap_df,
                                         width='rect_width', height='rect_height',
                                         fill_color=bktransform.log_cmap(what_to_plot[what_to_plot_cntr],
                                                                            bkpalettes.Plasma256,
                                                                            low=1,
                                                                            high=max(heatmap_df[what_to_plot[what_to_plot_cntr]])),
                                         line_width=0, line_alpha=0)
        # add colour bar
        figure.add_layout(rect.construct_color_bar(
            major_label_text_font_size="8pt",
            ticker=bkmodels.FixedTicker(ticks=which_ticks[what_to_plot_cntr]),
            formatter=bkmodels.PrintfTickFormatter(format="%d"),
            label_standoff=6,
            border_line_color=None,
            padding=5
        ), 'right')
        # plot the burden contours
        for i in range(0, len(xi_contours['values'])):
            for j in range(0, len(xi_contours['contour lines'][i])):
                figure.line(xi_contours['contour lines'][i][j][:, 0], xi_contours['contour lines'][i][j][:, 1],
                            line_dash=xi_contours['dashes'][i],
                            #legend_label=xi_contours['legends'][i],
                            line_width=2, line_color='black')

        # add and configure the legend
        # figure.legend.location = "bottom_left"
        # figure.legend.label_text_font_size = "8pt"
        # mark where the point with default parameters lays
        figure.x(x=[1/par['baseline_switch']], y=[par['eta_switch']], size=8, color='black', line_width=2)
        what_to_plot_cntr+=1

        # set fonts
        figure.xaxis.axis_label_text_font_size = "8pt"
        figure.xaxis.major_label_text_font_size = "8pt"
        figure.yaxis.axis_label_text_font_size = "8pt"
        figure.yaxis.major_label_text_font_size = "8pt"

        # add labels for burden thresholds
        figure.add_layout(bkmodels.Label(x=xi_contours['contour lines'][0][0][-1, 0],
                                         y=xi_contours['contour lines'][0][0][-1, 1],
                                         x_offset=16, y_offset=-32,
                                         text=xi_contours['legends'][0],
                                         text_font_size='8pt',
                                         text_color='white'))
        figure.add_layout(bkmodels.Label(x=xi_contours['contour lines'][1][0][-1, 0],
                                         y=xi_contours['contour lines'][1][0][-1, 1],
                                         x_offset=-40, y_offset=-48,
                                         text=xi_contours['legends'][1],
                                         text_font_size='8pt',
                                         text_color='white'))

    return gfchange_F_figure, gfchange_int_figure, gfchange_intact_figure


# MAKE THE PLOT FOR SWITCH AND INT GENE CONCS. AND PROPORTION OF INDUCER-BOUND P_SWITCH MOLECULES ----------------------
def c_acfrac_plot(c_range, acfrac_range,  # parameter ranges to consider
                  par, cellvars,  # model parameters and cellular variable values
                  xi_contours=None  # details of non-default burden contours to draw on the heatmap, if any
                  ):
    # get a mesh grid, then flatten its x and y coordinates into a single linear array
    c_mesh, acfrac_mesh = np.meshgrid(c_range, acfrac_range)
    c_mesh_ravel = c_mesh.ravel()
    acfrac_mesh_ravel = acfrac_mesh.ravel()

    # specify vmapping axes
    par_vmapping_axes = {}
    for key in par.keys():
        if ((key == 'c_switch' or key == 'c_int') or key == 'p_switch_ac_frac'):
            par_vmapping_axes[key] = 0
        else:
            par_vmapping_axes[key] = None
    cellvars_vmapping_axes = {}
    for key in cellvars.keys():
        if(key == 'xi_switch_max' or key == 'xi_int_max'):
            cellvars_vmapping_axes[key] = 0
        else:
            cellvars_vmapping_axes[key] = None

    # DETERMINE WHICH PARAMETER COMBINATIONS YIELD A DESIRED THRESHOLD BIFURCATION POINT -------------------------------
    # make a vmappable parameter dictionary
    par_for_existence = par.copy()
    par_for_existence['c_switch'] = jnp.array(c_mesh_ravel)
    par_for_existence['c_int'] = jnp.array(c_mesh_ravel)
    par_for_existence['p_switch_ac_frac'] = jnp.array(acfrac_mesh_ravel)

    # make a vmappable cellvars dictionary
    cellvars_for_existence = cellvars.copy()
    cellvars_for_existence['xi_switch_max'] = (cellvars['xi_switch_max']/par['c_switch']) * jnp.array(c_mesh_ravel)
    cellvars_for_existence['xi_int_max'] = (cellvars['xi_int_max']/par['c_int']) * jnp.array(c_mesh_ravel)


    # make the checking function vmappable
    vmapped_check_if_threshold_exists = jax.jit(jax.vmap(check_if_threshold_exists,
                                                         in_axes=(par_vmapping_axes, cellvars_vmapping_axes)))

    # run
    threshold_exists = np.array(vmapped_check_if_threshold_exists(par_for_existence, cellvars_for_existence))

    # FOR THOSE COMBINATIONS WHERE T/B POINT EXISTS, FIND IT AND THE GUARANTEED FOLD CHANGES ---------------------------
    # only consider parameter combinations where the threshold bifurcation point exists
    indices_where_threshold_exists = []
    for i in range(0, len(threshold_exists)):
        if (threshold_exists[i]):
            indices_where_threshold_exists.append(i)
    c_mesh_ravel_exists = c_mesh_ravel[indices_where_threshold_exists]
    acfrac_mesh_ravel_exists = acfrac_mesh_ravel[indices_where_threshold_exists]

    # make a vmappable parameter dictionary
    par_for_threshold_gfchanges = par.copy()
    par_for_threshold_gfchanges['c_switch'] = jnp.array(c_mesh_ravel_exists)
    par_for_threshold_gfchanges['c_int'] = jnp.array(c_mesh_ravel_exists)
    par_for_threshold_gfchanges['p_switch_ac_frac'] = jnp.array(acfrac_mesh_ravel_exists)

    # make a vmappable cellvars dictionary
    cellvars_for_threshold_gfchanges = cellvars.copy()
    cellvars_for_threshold_gfchanges['xi_switch_max'] = (cellvars['xi_switch_max']/par['c_switch']) * jnp.array(c_mesh_ravel_exists)
    cellvars_for_threshold_gfchanges['xi_int_max'] = (cellvars['xi_int_max']/par['c_int']) * jnp.array(c_mesh_ravel_exists)

    # make the threshold and guaranteed fold change retrieval function vmappable
    vmapped_threshold_gfchnages = jax.jit(jax.vmap(threshold_gfchnages,
                                                   in_axes=(par_vmapping_axes, cellvars_vmapping_axes)))

    # run
    thresholds_gfchanges = np.array(vmapped_threshold_gfchnages(par_for_threshold_gfchanges, cellvars_for_threshold_gfchanges))

    # unpack the results
    p_switch_thresholds = thresholds_gfchanges[:, 0]
    xi_thresholds = thresholds_gfchanges[:, 1]
    gfchange_ints = thresholds_gfchanges[:, 2]
    gfchange_Fs = thresholds_gfchanges[:, 3]
    gfchange_intacts = thresholds_gfchanges[:, 4]

    # FIND BURDEN CONTOURS  --------------------------------------------------------------------------------------------
    # fill the points where no threshold exists with INFS
    xi_thresholds_for_contour_ravel = np.zeros(threshold_exists.shape)  # initialise
    last_index_in_exist_list = 0
    for i in range(0, len(xi_thresholds_for_contour_ravel)):
        if (i == indices_where_threshold_exists[last_index_in_exist_list]):
            xi_thresholds_for_contour_ravel[i] = xi_thresholds[last_index_in_exist_list]
            if (last_index_in_exist_list < len(indices_where_threshold_exists) - 1):
                last_index_in_exist_list += 1
        else:
            xi_thresholds_for_contour_ravel[i] = np.inf
    xi_thresholds_for_contour = xi_thresholds_for_contour_ravel.reshape(len(c_range),
                                                                        len(acfrac_range)).T

    # create a contour generator
    threshold_cgen = cgen(x=acfrac_range, y=c_range,
                          z=xi_thresholds_for_contour)

    # if no xi values specified, plot one for all synthetic genes present and the one with just native genes and CAT
    if(xi_contours == None):
        xi_native_cat = cellvars['xi_a'] + cellvars['xi_r'] + cellvars['xi_cat']
        xi_with_all_genes = xi_native_cat + cellvars['xi_other_genes']
        xi_contours={'values':[xi_native_cat, xi_with_all_genes],
                     'legends': ['Native & CAT3 unles \n gene exp. burden', 'Total gene \n exp. burden'],
                     'dashes':['solid','dashed'],
                     'contour lines':[]}

    # find burden contour lines
    xi_contours['contour lines'] = []
    for i in range(0,len(xi_contours['values'])):
        xi_contours['contour lines'].append(threshold_cgen.lines(xi_contours['values'][i]))

    # PLOT THE RESULTS -------------------------------------------------------------------------------------------------
    # create output file
    # bkplot.output_file(filename='switching_behaviour_c_acfrac.html',
    #                    title='Gene conc., active fraction')

    # calculate widths and heights foatmap rectangles - unintuitive due to log scale
    rect_widths_along_x_axis = np.zeros(len(acfrac_range))
    rect_widths_along_x_axis[0] = acfrac_range[1] - acfrac_range[0]
    for i in range(1, len(acfrac_range)):
        rect_widths_along_x_axis[i] = ((acfrac_range[i] - acfrac_range[i - 1]) -
                                       rect_widths_along_x_axis[i - 1] / 2) * 2
    rect_heights_along_y_axis = np.zeros(len(c_range))
    rect_heights_along_y_axis[0] = c_range[1] - c_range[0]
    for i in range(1, len(c_range)):
        rect_heights_along_y_axis[i] = ((c_range[i] - c_range[i - 1]) - rect_heights_along_y_axis[i - 1] / 2) * 2
    rect_widths_ravel_exists = np.zeros(acfrac_mesh_ravel_exists.shape)
    rect_heights_ravel_exists = np.zeros(c_mesh_ravel_exists.shape)
    for i in range(0, len(acfrac_mesh_ravel_exists)):
        baseline_where = np.argwhere(
            acfrac_range == acfrac_mesh_ravel_exists[i])  # locate the baseline value in the baseline range
        rect_widths_ravel_exists[i] = rect_widths_along_x_axis[baseline_where[0][0]]*1.25
        eta_where = np.argwhere(c_range == c_mesh_ravel_exists[i])  # locate the eta value in the eta range
        rect_heights_ravel_exists[i] = rect_heights_along_y_axis[eta_where[0][0]]*1.25

    # make a dataframe for the heatmap of guaranteed fold changes
    heatmap_df = pd.DataFrame({'c_s=c_int': c_mesh_ravel_exists, 'I': acfrac_mesh_ravel_exists,
                               'gfchange_F':gfchange_Fs, 'gfchange_int': gfchange_ints, 'gfchange_intact': gfchange_intacts,
                               'rect_width': rect_widths_ravel_exists, 'rect_height': rect_heights_ravel_exists})

    # PLOT guaranteed fold-changes
    gfchange_F_figure = bkplot.figure(
        frame_width=240,
        frame_height=180,
        x_axis_label="Share of switch proteins bound by inducer",
        y_axis_label="Switch and int. gene conc., nM",
        x_range=(min(acfrac_range), max(acfrac_range)),
        y_range=(min(c_range), max(c_range)),
        #title="F_switch value GF-change",
        tools='pan,box_zoom,reset,save'
    )
    gfchange_int_figure = bkplot.figure(
        frame_width=240,
        frame_height=180,
        x_axis_label="Share of switch proteins bound by inducer",
        y_axis_label="Switch and int. gene conc., nM",
        x_range=(min(acfrac_range), max(acfrac_range)),
        y_range=(min(c_range), max(c_range)),
        #title="Integrase expression GF-change",
        tools='pan,box_zoom,reset,save'
    )
    gfchange_intact_figure = bkplot.figure(
        frame_width=240,
        frame_height=180,
        x_axis_label="Share of switch proteins bound by inducer",
        y_axis_label="Switch and int. gene conc., nM",
        x_range=(min(acfrac_range), max(acfrac_range)),
        y_range=(min(c_range), max(c_range)),
        #title="Integrase activity GF-change",
        tools='pan,box_zoom,reset,save'
    )
    what_to_plot = ['gfchange_F', 'gfchange_int', 'gfchange_intact']
    what_to_plot_cntr = 0
    for figure in [gfchange_F_figure, gfchange_int_figure, gfchange_intact_figure]:
        # svg backend
        figure.output_backend= "svg"
        # specify the colour scale and ticks
        if(what_to_plot[what_to_plot_cntr] == 'gfchange_intact'):
            colourscale_min=1
            which_ticks=np.power(10,(np.arange(0,np.log10(max(heatmap_df[what_to_plot[what_to_plot_cntr]])),1)))
            tick_format='%d'
        else:
            colourscale_min=min(heatmap_df[what_to_plot[what_to_plot_cntr]])
            which_ticks=np.logspace(np.log10(colourscale_min),np.log10(max(heatmap_df[what_to_plot[what_to_plot_cntr]])),4)
            tick_format='%.2f'
        # plot the heatmap
        rects = figure.rect(x="I", y="c_s=c_int", source=heatmap_df,
                                         width='rect_width', height='rect_height',
                                         fill_color=bktransform.log_cmap(what_to_plot[what_to_plot_cntr],
                                                                            bkpalettes.Plasma256,
                                                                            low=colourscale_min,
                                                                            high=max(heatmap_df[what_to_plot[what_to_plot_cntr]])),
                                         line_width=0,line_alpha=0)
        # add colour bar
        figure.add_layout(rects.construct_color_bar(
            major_label_text_font_size="8pt",
            ticker=bkmodels.FixedTicker(ticks=which_ticks),
            formatter=bkmodels.PrintfTickFormatter(format=tick_format),
            label_standoff=6,
            border_line_color=None,
            padding=5
        ), 'right')

        # mark the line of same DNA concentrations as the default parameter
        figure.line(x=[min(acfrac_range), max(acfrac_range)], y=[par['c_switch'], par['c_switch']],
                    line_width=2, line_color='white', line_dash='solid')

        # plot the burden contours
        for i in range(0,len(xi_contours['values'])):
            for j in range(0,len(xi_contours['contour lines'][i])):
                figure.line(xi_contours['contour lines'][i][j][:, 0], xi_contours['contour lines'][i][j][:, 1],
                            line_dash=xi_contours['dashes'][i],
                            #legend_label=xi_contours['legends'][i],
                            line_width=2, line_color='black')

        # add and configure the legend
        # figure.legend.location = "bottom_left"
        # figure.legend.label_text_font_size = "8pt"

        # mark where the point with default parameters lays
        figure.x(x=[par['p_switch_ac_frac']], y=[par['c_int']], size=10, color='black',line_width=4)

        # add burden contour labels
        figure.add_layout(bkmodels.Label(x=xi_contours['contour lines'][0][0][0, 0],
                                         y=xi_contours['contour lines'][0][0][0, 1],
                                         x_offset=-80, y_offset=-32,
                                         text=xi_contours['legends'][0],
                                         text_font_size='8pt',
                                         text_color='white'))
        figure.add_layout(bkmodels.Label(x=xi_contours['contour lines'][1][0][0, 0],
                                         y=xi_contours['contour lines'][1][0][0, 1],
                                         x_offset=24, y_offset=-32,
                                         text=xi_contours['legends'][1],
                                         text_font_size='8pt',
                                         text_color='white'))

        # font size
        figure.xaxis.axis_label_text_font_size = "8pt"
        figure.xaxis.major_label_text_font_size = "8pt"
        figure.yaxis.axis_label_text_font_size = "8pt"
        figure.yaxis.major_label_text_font_size = "8pt"

        what_to_plot_cntr += 1

    # PLOT how inducer concentration can be used to tune the switching behaviour
    # find the slice of data for the default gene concentration
    slice_ind = np.argmin(np.abs(c_range - par['c_switch']))
    tuning_figure = bkplot.figure(
        frame_width=240,
        frame_height=180,
        x_axis_label="Share of switch proteins bound by inducer",
        y_axis_label="Threshold gen exp. burden",
        x_range=(min(acfrac_range), max(acfrac_range)),
        y_range=(min(xi_thresholds_for_contour[slice_ind, :])*0.75, max(xi_thresholds_for_contour[slice_ind, :])*1.25),
        #title="Integrase activity GF-change",
        tools='pan,box_zoom,reset,save'
    )
    # svg backend
    tuning_figure.output_backend= "svg"

    # plot the threshold burden values
    tuning_figure.line(x=acfrac_range, y=xi_thresholds_for_contour[slice_ind, :],
                          line_width=2, line_color=bkRGB(255, 103, 0), line_dash='solid')

    return gfchange_F_figure,gfchange_int_figure,gfchange_intact_figure, tuning_figure


# STUDY THE SWITCHING BEHAVIOUR OF THE PUNISHER WITH A SINGLE EXTRA BURDENSOME GENE ------------------------------------
def switching_for_punisher_xtra():
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
    par['a_xtra'] = 1000.0

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

    # get the values of the variables assumed to be constant
    simulate_to_find_cellvars = True  # get these values by simulation (True) or load from a saved file (False)
    if (simulate_to_find_cellvars):
        # get the cellular variables
        e, F_r, h, xi_a, xi_r, xi_cat, xi_switch_max, x_int_max, xi_other_genes = val4an(par, ode_with_circuit, init_conds,
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

    # MAKE THE PLOT FOR ETA AND RECIPROCAL OF BASELINE F_SWITCH VALUE --------------------------------------------------
    # define parameter ranges
    eta_range=np.linspace(1.01,4,200)
    reciprocal_baseline_range=np.logspace(np.log10(1),np.log10(200),200)

    # define the output file for the plots
    bkplot.output_file(filename='switching_behaviour_eta_baseline.html',
                       title='Switching\'s depenedence on eta and baseline')

    # make the plots
    gfchange_F_figure, gfchange_int_figure, gfchange_intact_figure = eta_reciprocalbaseline_plot(eta_range, reciprocal_baseline_range,par, cellvars)

    # save plots
    bkplot.save(bklayouts.grid([[gfchange_F_figure, gfchange_int_figure, gfchange_intact_figure]]))

    # MAKE THE PLOT FOR SWITCH AND INT GENE CONCS. AND ACTIVAE FRACTION OF P_SWITCH ------------------------------------
    # define parameter ranges
    c_range=np.linspace(5,20,75)
    ac_frac_range=np.linspace(0.01,1,75)

    # define the output file for the plots
    bkplot.output_file(filename='switching_behaviour_c_acfrac.html',
                       title='Gene conc., active fraction')

    # make the plots
    gfchange_F_figure, gfchange_int_figure, gfchange_intact_figure, tuning_figure = c_acfrac_plot(c_range, ac_frac_range,par, cellvars)

    # save plots
    bkplot.save(bklayouts.grid([[gfchange_F_figure, gfchange_int_figure, gfchange_intact_figure],
                                [tuning_figure, None, None]]))



    # MAKE A SINGLE CASE INVESTIGATION FOR THE DEFAULT PARAMETER VALUES -------------------------------------------------
    if not (check_if_threshold_exists(par,cellvars)):
        print('No threshold bifurcation exists')
        return

    # find the threshold values and guaranteed fold changes
    thresholds_gfchanges = np.array(threshold_gfchnages(par, cellvars))
    p_switch_threshold = thresholds_gfchanges[0]
    xi_threshold = thresholds_gfchanges[1]
    gfchange_int = thresholds_gfchanges[2]
    gfchange_F = thresholds_gfchanges[3]
    gfchange_intact = thresholds_gfchanges[4]

    # plot
    plots_for_single_case(p_switch_threshold,xi_threshold,gfchange_F,par,cellvars)

    return


# STUDY THE SWITCHING BEHAVIOUR OF THE PUNISHER WITH TWO TOGGLE SWITCHES -----------------------------------------------
def switching_for_twotoggles_punisher():
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
        init_conds['m_tog' + str(togswitchnum) + '1'] = 500

    # Punisher parameters
    par['c_switch'] = 10.0  # gene concentration (nM)
    par['a_switch'] = 100.0  # promoter strength (unitless)
    par['c_int'] = 10.0  # gene concentration (nM)
    par['a_int'] = 60.0  # promoter strength (unitless)
    par['d_int'] = 6.0  # integrase protein degradation rate (to avoid unnecessary punishment)
    par['c_cat'] = 10.0  # gene concentration (nM)
    par['a_cat'] = 500.0  # promoter strength (unitless)

    # punisher's transcription regulation function
    par['K_switch'] = 600.0  # Half-saturation constant for the self-activating switch gene promoter (nM)
    par['eta_switch'] = 2

    # Hill coefficient for the self-activating switch gene promoter (unitless)
    par['baseline_switch'] = 0.025  # Baseline value of the switch gene's transcription activation function
    par['p_switch_ac_frac'] = 0.8  # active fraction of protein (i.e. share of molecules bound by the inducer)

    # culture medium
    init_conds['s'] = 0.3
    par['h_ext'] = 10.5 * (10.0 ** 3)

    # get the values of the variables assumed to be constant
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
    # DEFINE WHICH BURDEN CONTOURS TO PLOT -----------------------------------------------------------------------------
    # get the burden with all synthetic genes present
    xi_all_present = cellvars['xi_a'] + cellvars['xi_r'] + cellvars['xi_cat'] +cellvars['xi_other_genes']
    # also get the burden with just one of the toggle switches functional
    # (here we care about losing a single toggle switch more than about eventually losing both)
    par_loseonetoggle=par.copy()
    par_loseonetoggle['func_tog11'] = 0
    par_loseonetoggle['func_tog12'] = 0
    _, _, _, _, _, _, _, _, xi_other_genes_justonetoggle = val4an(par_loseonetoggle, ode_with_circuit,
                                                                  init_conds,
                                                                  circuit_genes, circuit_miscs,
                                                                  circuit_name2pos,
                                                                  circuit_F_calc)
    xi_justonetoggle = cellvars['xi_a'] + cellvars['xi_r'] + cellvars['xi_cat']+ xi_other_genes_justonetoggle

    # specify the legend entries and dashings for burden contours
    xi_contours = {'values':[xi_justonetoggle,xi_all_present],
                   'legends':['One toggle','Both toggles'],
                   'dashes':['solid','dashed'],
                   'contour lines': []}

    # MAKE THE PLOT FOR ETA AND RECIPROCAL OF BASELINE F_SWITCH VALUE --------------------------------------------------
    # define parameter ranges
    eta_range = np.linspace(1.01, 4, 300)
    reciprocal_baseline_range = np.logspace(np.log10(1), np.log10(200), 300)

    # define the output file for the plots
    bkplot.output_file(filename='switching_behaviour_eta_baseline.html',
                       title='Switching\'s depenedence on eta and baseline')

    # make the plots
    gfchange_F_figure, gfchange_int_figure, gfchange_intact_figure = eta_reciprocalbaseline_plot(eta_range,
                                                                                                 reciprocal_baseline_range,
                                                                                                 par, cellvars,
                                                                                                 xi_contours.copy())

    # save plots
    bkplot.save(bklayouts.grid([[gfchange_F_figure, gfchange_int_figure, gfchange_intact_figure]]))

    # MAKE THE PLOT FOR SWITCH AND INT GENE CONCS. AND ACTIVAE FRACTION OF P_SWITCH ------------------------------------
    # define parameter ranges
    c_range = np.linspace(5, 20, 75)
    ac_frac_range = np.linspace(0.01, 1, 75)

    # define the output file for the plots
    bkplot.output_file(filename='switching_behaviour_c_acfrac.html',
                       title='Gene conc., active fraction')

    # make the plots
    gfchange_F_figure, gfchange_int_figure, gfchange_intact_figure, tuning_figure = c_acfrac_plot(c_range, ac_frac_range,
                                                                                   par, cellvars,
                                                                                   xi_contours.copy())

    # save plots
    bkplot.save(bklayouts.grid([[gfchange_F_figure, gfchange_int_figure, gfchange_intact_figure]]))

    # MAKE A SINGLE CASE INVESTIGATION FOR THE DEFAULT PARAMETER VALUES -------------------------------------------------
    if not (check_if_threshold_exists(par, cellvars)):
        print('No threshold bifurcation exists')
        return

    # find the threshold values and guaranteed fold changes
    thresholds_gfchanges = np.array(threshold_gfchnages(par, cellvars))
    p_switch_threshold = thresholds_gfchanges[0]
    xi_threshold = thresholds_gfchanges[1]
    gfchange_int = thresholds_gfchanges[2]
    gfchange_F = thresholds_gfchanges[3]
    gfchange_intact = thresholds_gfchanges[4]

    # plot
    plots_for_single_case(p_switch_threshold, xi_threshold, gfchange_F, par, cellvars)

    return

# MAIN FUNCTION --------------------------------------------------------------------------------------------------------
def main():
    #switching_for_twotoggles_punisher()
    switching_for_punisher_xtra()
    return


# MAIN CALL ------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
