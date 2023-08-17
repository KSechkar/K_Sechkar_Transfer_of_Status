'''
SYNTHETIC_CIRCUITS_JAX.PY: Describing different synthetic gene circuits
for the Python/Jax implementation of the coarse-grained resource-aware E.coli model

Redone to enable fast jax.jit compilation
'''

import jax.numpy as jnp
import jax.lax as lax

# NO SYNTHETIC GENES ---------------------------------------------------------------------------------------------------
# initialise all the necessary parameters to simulate the circuit
# return the default parameters and initial conditions, species name to ODE vector position decoder and plot colour palette
def nocircuit_initialise():
    default_par={'cat_gene_present':0} # chloramphenicol resistance gene present
    default_init_conds={}
    genes={}
    miscs={}
    name2pos={'p_cat':0} # placeholder, will never be used but required for correct execution'}
    circuit_colours={}
    return default_par, default_init_conds, genes, miscs, name2pos, circuit_colours

# transcription regulation functions
def nocircuit_F_calc(t ,x, par, name2pos):
    return jnp.array([])

# ode
def nocircuit_ode(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state, external inputs
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            par,  # system parameters
            name2pos  # name to position decoder
            ):
    # RETURN THE ODE
    return []



# ONE CONSTITUTIVE GENE ------------------------------------------------------------------------------------------------
# initialise all the necessary parameters to simulate the circuit
# return the default parameters and initial conditions, species name to ODE vector position decoder and plot colour palette
def oneconstitutive_initialise():
    # -------- SPECIFY CIRCUIT COMPONENTS FROM HERE...
    genes = ['xtra']  # names of genes in the circuit
    miscs = []  # names of miscellaneous species involved in the circuit (e.g. metabolites)
    # -------- ...TO HERE

    # for convenience, one can refer to the species' concs. by names instead of positions in x
    # e.g. x[name2pos['m_xtra']] will return the concentration of mRNA of the gene 'xtra'
    name2pos = {}
    for i in range(0, len(genes)):
        name2pos['m_' + genes[i]] = 8 + i  # mRNA
        name2pos['p_' + genes[i]] = 8 + len(genes) + i  # protein
    for i in range(0, len(miscs)):
        name2pos[miscs[i]] = 8 + len(genes) * 2 + i  # miscellaneous species
    for i in range(0, len(genes)):
        name2pos['k_' + genes[i]] =  i  # effective mRNA-ribosome dissociation constants (in k_het, not x!!!)
    for i in range(0, len(genes)):
        name2pos['F_' + genes[i]] =  i  # transcription regulation functions (in F, not x!!!)

    # default gene parameters to be imported into the main model's parameter dictionary
    default_par = {}
    for gene in genes: # gene parameters
        default_par['func_' + gene] = 1.0  # gene functionality - 1 if working, 0 if mutated
        default_par['c_' + gene] = 1.0  # copy no. (nM)
        default_par['a_' + gene] = 100.0  # promoter strength (unitless)
        default_par['b_' + gene] = 6.0  # mRNA decay rate (/h)
        default_par['k+_' + gene] = 60.0  # ribosome binding rate (/h/nM)
        default_par['k-_' + gene] = 60.0  # ribosome unbinding rate (/h)
        default_par['n_' + gene] = 300.0  # protein length (aa)
        default_par['d_' + gene] = 0.0  # rate of active protein degradation - zero by default (/h)

    # chloramphenicol resistance
    if ('cat' in genes):
        default_par['cat_gene_present'] = 1  # chloramphenicol resistance gene present
    else:
        default_par['cat_gene_present'] = 0  # chloramphenicol resistance gene absent
        name2pos['p_cat']=0 # placeholder, will never be used but required for correct execution

    # default initial conditions
    default_init_conds = {}
    for gene in genes:
        default_init_conds['m_' + gene] = 0
        default_init_conds['p_' + gene] = 0
    for misc in miscs:
        default_init_conds[misc] = 0

    # -------- DEFAULT VALUES OF CIRCUIT-SPECIFIC PARAMETERS CAN BE SPECIFIED FROM HERE...
    # -------- ...TO HERE

    # default palette and dashes for plotting (5 genes + misc. species max)
    default_palette = ["#0072BD", "#D95319", "#4DBEEE", "#A2142F", "#FF00FF"]
    default_dash = ['solid']
    # match default palette to genes and miscellaneous species, looping over the five colours we defined
    circuit_styles = {'colours': {}, 'dashes': {}}  # initialise dictionary
    for i in range(0, len(genes)):
        circuit_styles['colours'][genes[i]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][genes[i]] = default_dash[i % len(default_dash)]
    for i in range(len(genes), len(genes) + len(miscs)):
        circuit_styles['colours'][miscs[i - len(genes)]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][miscs[i - len(genes)]] = default_dash[i % len(default_dash)]

    # --------  YOU CAN RE-SPECIFY COLOURS FOR PLOTTING FROM HERE...
    # -------- ...TO HERE

    return default_par, default_init_conds, genes, miscs, name2pos, circuit_styles

# transcription regulation functions
def oneconstitutive_F_calc(t ,x, par, name2pos):
    F_xtra = 1 # constitutive gene
    return jnp.array([F_xtra])

# ode
def oneconstitutive_ode(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state, external inputs
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            par,  # system parameters
            name2pos  # name to position decoder
            ):
    # GET REGULATORY FUNCTION VALUES
    F = F_calc(t, x, par, name2pos)

    # RETURN THE ODE
    return [# mRNAs
            par['func_xtra'] * l * F[name2pos['F_xtra']] * par['c_xtra'] * par['a_xtra'] - (par['b_xtra'] + l) * x[name2pos['m_xtra']],
            # proteins
            (e / par['n_xtra']) * (x[name2pos['m_xtra']] / k_het[name2pos['k_xtra']] / D) * R - (l + par['d_xtra']) * x[name2pos['p_xtra']]
    ]

# ONE CONSTITUTIVE GENE + CHLORAMPHENICOL RESISTANCE--------------------------------------------------------------------
# initialise all the necessary parameters to simulate the circuit
def oneconstitutive_cat_initialise():
    # -------- SPECIFY CIRCUIT COMPONENTS FROM HERE...
    genes = ['xtra','cat']  # names of genes in the circuit
    miscs = []  # names of miscellaneous species involved in the circuit (e.g. metabolites)
    # -------- ...TO HERE

    # for convenience, one can refer to the species' concs. by names instead of positions in x
    # e.g. x[name2pos['m_xtra']] will return the concentration of mRNA of the gene 'xtra'
    name2pos = {}
    for i in range(0, len(genes)):
        name2pos['m_' + genes[i]] = 8 + i  # mRNA
        name2pos['p_' + genes[i]] = 8 + len(genes) + i  # protein
    for i in range(0, len(miscs)):
        name2pos[miscs[i]] = 8 + len(genes) * 2 + i  # miscellaneous species
    for i in range(0, len(genes)):
        name2pos['k_' + genes[i]] =  i  # effective mRNA-ribosome dissociation constants (in k_het, not x!!!)
    for i in range(0, len(genes)):
        name2pos['F_' + genes[i]] = i  # transcription regulation functions (in F, not x!!!)

    # default gene parameters to be imported into the main model's parameter dictionary
    default_par = {}
    for gene in genes: # gene parameters
        default_par['func_' + gene] = 1.0  # gene functionality - 1 if working, 0 if mutated
        default_par['c_' + gene] = 1.0  # copy no. (nM)
        default_par['a_' + gene] = 100.0  # promoter strength (unitless)
        default_par['b_' + gene] = 6.0  # mRNA decay rate (/h)
        default_par['k+_' + gene] = 60.0  # ribosome binding rate (/h/nM)
        default_par['k-_' + gene] = 60.0  # ribosome unbinding rate (/h)
        default_par['n_' + gene] = 300.0  # protein length (aa)
        default_par['d_' + gene] = 0.0  # rate of active protein degradation - zero by default (/h)

    # chloramphenicol resistance
    if ('cat' in genes):
        default_par['cat_gene_present'] = 1  # chloramphenicol resistance gene present
    else:
        default_par['cat_gene_present'] = 0  # chloramphenicol resistance gene absent
        name2pos['p_cat']=0 # placeholder, will never be used but required for correct execution

    # default initial conditions
    default_init_conds = {}
    for gene in genes:
        default_init_conds['m_' + gene] = 0
        default_init_conds['p_' + gene] = 0
    for misc in miscs:
        default_init_conds[misc] = 0

    # -------- DEFAULT VALUES OF CIRCUIT-SPECIFIC PARAMETERS CAN BE SPECIFIED FROM HERE...
    # -------- ...TO HERE

    # default palette and dashes for plotting (5 genes + misc. species max)
    default_palette = ["#0072BD", "#D95319", "#4DBEEE", "#A2142F", "#FF00FF"]
    default_dash = ['solid']
    # match default palette to genes and miscellaneous species, looping over the five colours we defined
    circuit_styles={'colours':{}, 'dashes':{}} # initialise dictionary
    for i in range(0, len(genes)):
        circuit_styles['colours'][genes[i]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][genes[i]] = default_dash[i % len(default_dash)]
    for i in range(len(genes), len(genes) + len(miscs)):
        circuit_styles['colours'][miscs[i - len(genes)]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][miscs[i - len(genes)]] = default_dash[i % len(default_dash)]

    # --------  YOU CAN RE-SPECIFY COLOURS FOR PLOTTING FROM HERE...
    # -------- ...TO HERE

    return default_par, default_init_conds, genes, miscs, name2pos, circuit_styles

# transcription regulation functions
def oneconstitutive_cat_F_calc(t ,x, par, name2pos):
    F_xtra = 1 # constitutive gene
    F_cat = 1 # constitutive gene
    return jnp.array([F_xtra, F_cat])

# ode
def oneconstitutive_cat_ode(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state, external inputs
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            par,  # system parameters
            name2pos  # name to position decoder
            ):
    # GET REGULATORY FUNCTION VALUES
    F = F_calc(t, x, par, name2pos)

    # RETURN THE ODE
    return [# mRNAs
            par['func_xtra'] * l * F[name2pos['F_xtra']] * par['c_xtra'] * par['a_xtra'] - (par['b_xtra'] + l) * x[name2pos['m_xtra']],
            par['func_cat'] * l * F[name2pos['F_cat']] * par['c_cat'] * par['a_cat'] - (par['b_cat'] + l) * x[name2pos['m_cat']],
            # proteins
            (e / par['n_xtra']) * (x[name2pos['m_xtra']] / k_het[name2pos['k_xtra']] / D) * R - (l + par['d_xtra']) * x[name2pos['p_xtra']],
            (e / par['n_cat']) * (x[name2pos['m_cat']] / k_het[name2pos['k_cat']] / D) * R - (l + par['d_cat']) * x[name2pos['p_cat']]
    ]

# PUNISHER AND A SINGLE BURDENSOME GENE --------------------------------------------------------------------------------
# initialise all the necessary parameters to simulate the circuit
def punisher_xtra_initialise():
    # -------- SPECIFY CIRCUIT COMPONENTS FROM HERE...
    genes = ['xtra',
             'switch', 'int', 'cat']  # names of genes in the circuit
    miscs = ['cat_pb', 'cat_lri1']  # names of miscellaneous species involved in the circuit. Here, cat gene states with respect to the integrase
    # -------- ...TO HERE

    # for convenience, one can refer to the species' concs. by names instead of positions in x
    # e.g. x[name2pos['m_xtra']] will return the concentration of mRNA of the gene 'xtra'
    name2pos = {}
    for i in range(0, len(genes)):
        name2pos['m_' + genes[i]] = 8 + i  # mRNA
        name2pos['p_' + genes[i]] = 8 + len(genes) + i  # protein
    for i in range(0, len(miscs)):
        name2pos[miscs[i]] = 8 + len(genes) * 2 + i  # miscellaneous species
    for i in range(0, len(genes)):
        name2pos['k_' + genes[i]] =  i  # effective mRNA-ribosome dissociation constants (in k_het, not x!!!)
    for i in range(0, len(genes)):
        name2pos['F_' + genes[i]] = i  # transcription regulation functions (in F, not x!!!)

    # default gene parameters to be imported into the main model's parameter dictionary
    default_par = {}
    for gene in genes: # gene parameters
        default_par['func_' + gene] = 1.0  # gene functionality - 1 if working, 0 if mutated
        default_par['c_' + gene] = 1.0  # copy no. (nM)
        default_par['a_' + gene] = 100.0  # promoter strength (unitless)
        default_par['b_' + gene] = 6.0  # mRNA decay rate (/h)
        default_par['k+_' + gene] = 60.0  # ribosome binding rate (/h/nM)
        default_par['k-_' + gene] = 60.0  # ribosome unbinding rate (/h)
        default_par['n_' + gene] = 300.0  # protein length (aa)
        default_par['d_' + gene] = 0.0  # rate of active protein degradation - zero by default (/h)

    # chloramphenicol resistance
    if ('cat' in genes):
        default_par['cat_gene_present'] = 1  # chloramphenicol resistance gene present
    else:
        default_par['cat_gene_present'] = 0  # chloramphenicol resistance gene absent
        name2pos['p_cat']=0 # placeholder, will never be used but required for correct execution

    # default initial conditions
    default_init_conds = {}
    for gene in genes:
        default_init_conds['m_' + gene] = 0
        default_init_conds['p_' + gene] = 0
    for misc in miscs:
        default_init_conds[misc] = 0

    # -------- DEFAULT VALUES OF CIRCUIT-SPECIFIC PARAMETERS CAN BE SPECIFIED FROM HERE...
    # punisher
    default_par['c_switch'] = 1  # gene concentration (nM)
    default_par['a_switch'] = 150  # promoter strength (unitless)
    default_par['c_int'] = 1  # gene concentration (nM)
    default_par['a_int'] = 1000  # promoter strength (unitless)
    default_par['d_int'] = 6  # integrase protein degradation rate (to avoid unnecessary punishment)
    default_par['c_cat'] = 1  # gene concentration (nM)
    default_par['a_cat'] = 5000  # promoter strength (unitless)

    # transcription regulation function
    default_par['K_switch'] = 100
    default_par['eta_switch'] = 2
    default_par['baseline_switch'] = 0.1
    default_par['p_switch_ac_frac'] = 1  # ACTIVE fraction of protein (i.e. share of molecules bound by the inducer)

    # integrase action
    default_par['K_bI~'] = 100 # dissociation constant for the integrase and DNA (nM)
    default_par['k_sxf'] = 6 # forward strain exchange rate by the integrase(1/h)
    default_par['k_sxr'] = 2.14 # reverse strain exchange rate by the integrase(1/h)
    default_par['k_+syn'] = 0.006 # forward synaptic conformational change rate (1/h)

    # initial conditions for the state of cat gene with respect to the integrase
    default_init_conds['cat_pb'] = 10 # originally, the gene is in functional state (with attP and attB sites)
    default_init_conds['cat_lri1'] = 0 # originally, the gene is in functional state (with attP and attB sites)
    # -------- ...TO HERE

    # default palette and dashes for plotting (5 genes + misc. species max)
    default_palette = ["#0072BD", "#D95319", "#4DBEEE", "#A2142F", "#FF00FF"]
    default_dash = ['solid']
    # match default palette to genes and miscellaneous species, looping over the five colours we defined
    circuit_styles={'colours':{}, 'dashes':{}} # initialise dictionary
    for i in range(0, len(genes)):
        circuit_styles['colours'][genes[i]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][genes[i]] = default_dash[i % len(default_dash)]
    for i in range(len(genes), len(genes) + len(miscs)):
        circuit_styles['colours'][miscs[i - len(genes)]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][miscs[i - len(genes)]] = default_dash[i % len(default_dash)]

    # --------  YOU CAN RE-SPECIFY COLOURS FOR PLOTTING FROM HERE...
    # give cat gene states same colours as cat mRNA and protein but different dashes
    circuit_styles['colours']['cat_pb'] = circuit_styles['colours']['cat']
    circuit_styles['colours']['cat_lri1'] = circuit_styles['colours']['cat']

    circuit_styles['dashes']['cat_pb'] = 'solid' # fully functional gene => solid dash
    circuit_styles['dashes']['cat_lri1'] = 'dashed' # non-functional => non-solid
    # -------- ...TO HERE

    return default_par, default_init_conds, genes, miscs, name2pos, circuit_styles

# transcription regulation functions
def punisher_xtra_F_calc(t ,x, par, name2pos):
    F_xtra = 1 # constitutive gene
    F_cat = 1 # constitutive gene

    # switch and integrase genes regulated by p_switch
    p_switch_dependent_term = (x[name2pos['p_switch']]*par['p_switch_ac_frac']/par['K_switch'])**par['eta_switch']
    F_switch = par['baseline_switch'] + (1 - par['baseline_switch']) * (p_switch_dependent_term/(p_switch_dependent_term+1))
    return jnp.array([F_xtra,
            F_switch,
            F_switch,  # integrase co-regulated with the switch gene
            F_cat])

# ode
def punisher_xtra_ode(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state, external inputs
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            par,  # system parameters
            name2pos  # name to position decoder
            ):
    # GET REGULATORY FUNCTION VALUES
    F = F_calc(t, x, par, name2pos)

    # get the Hill equation for integrase-cat DNA binding
    int_cat_hill = (x[name2pos['p_int']]/par['K_bI~'])**4/(1+(x[name2pos['p_int']]/par['K_bI~'])**4)

    # RETURN THE ODE
    return [# mRNAs
            par['func_xtra'] * l * F[name2pos['F_xtra']] * par['c_xtra'] * par['a_xtra'] - (par['b_xtra'] + l) * x[name2pos['m_xtra']],
            par['func_switch'] * l * F[name2pos['F_switch']] * par['c_switch'] * par['a_switch'] - (par['b_switch'] + l) * x[name2pos['m_switch']],
            par['func_int'] * l * F[name2pos['F_int']] * par['c_int'] * par['a_int'] - (par['b_int'] + l) * x[name2pos['m_int']],
            par['func_cat'] * l * F[name2pos['F_cat']] * x[name2pos['cat_pb']] * par['a_cat'] - (par['b_cat'] + l) * x[name2pos['m_cat']],  # NOTE: GENE COPY NO. GIVEN BY CMR_PB!
            # proteins
            (e / par['n_xtra']) * (x[name2pos['m_xtra']] / k_het[name2pos['k_xtra']] / D) * R - (l + par['d_xtra']) * x[name2pos['p_xtra']],
            (e / par['n_switch']) * (x[name2pos['m_switch']] / k_het[name2pos['k_switch']] / D) * R - (l + par['d_switch']) * x[name2pos['p_switch']],
            (e / par['n_int']) * (x[name2pos['m_int']] / k_het[name2pos['k_int']] / D) * R - (l + par['d_int']) * x[name2pos['p_int']],
            (e / par['n_cat']) * (x[name2pos['m_cat']] / k_het[name2pos['k_cat']] / D) * R - (l + par['d_cat']) * x[name2pos['p_cat']],
            # cat gene state with regard to the integrase
            # PB/PBI: fully functional (with attP and attB sites) free or with integrase bound
            -par['k_sxf']*int_cat_hill*x[name2pos['cat_pb']]+par['k_sxr']*x[name2pos['cat_lri1']],
            # LRI1: non-functional (with attL and attR sites) with integrase bound in active conformation - strands synapsed together
            (par['k_sxf'] * int_cat_hill * x[name2pos['cat_pb']] - par['k_sxr'] * x[name2pos['cat_lri1']]) + (
                -par['k_+syn'] * x[name2pos['cat_lri1']] - l*x[name2pos['cat_lri1']]) # no replenishment from LRI2/LR as strands diffuse away
    ]

# TWO TOGGLE SWITCHES AND A PUNISHER -----------------------------------------------------------------------------------
def twotoggles_punisher_initialise():
    # -------- SPECIFY CIRCUIT COMPONENTS FROM HERE...
    genes = ['tog11','tog12','tog21','tog22',  # toggle switches
             'switch', 'int', 'cat']  # punisher
    miscs = ['cat_pb', 'cat_lri1']  # names of miscellaneous species involved in the circuit. Here, cat gene states with respect to the integrase
    # -------- ...TO HERE

    # for convenience, one can refer to the species' concs. by names instead of positions in x
    # e.g. x[name2pos['m_xtra']] will return the concentration of mRNA of the gene 'xtra'
    name2pos = {}
    for i in range(0, len(genes)):
        name2pos['m_' + genes[i]] = 8 + i  # mRNA
        name2pos['p_' + genes[i]] = 8 + len(genes) + i  # protein
    for i in range(0, len(miscs)):
        name2pos[miscs[i]] = 8 + len(genes) * 2 + i  # miscellaneous species
    for i in range(0, len(genes)):
        name2pos['k_' + genes[i]] = i  # effective mRNA-ribosome dissociation constants (in k_het, not x!!!)
    for i in range(0, len(genes)):
        name2pos['F_' + genes[i]] = i  # transcription regulation functions (in F, not x!!!)

    # default gene parameters to be imported into the main model's parameter dictionary
    default_par = {}
    for gene in genes:  # gene parameters
        default_par['func_' + gene] = 1.0  # gene functionality - 1 if working, 0 if mutated
        default_par['c_' + gene] = 1.0  # copy no. (nM)
        default_par['a_' + gene] = 100.0  # promoter strength (unitless)
        default_par['b_' + gene] = 6.0  # mRNA decay rate (/h)
        default_par['k+_' + gene] = 60.0  # ribosome binding rate (/h/nM)
        default_par['k-_' + gene] = 60.0  # ribosome unbinding rate (/h)
        default_par['n_' + gene] = 300.0  # protein length (aa)
        default_par['d_' + gene] = 0.0  # rate of active protein degradation - zero by default (/h)

    # chloramphenicol resistance
    if ('cat' in genes):
        default_par['cat_gene_present'] = 1  # chloramphenicol resistance gene present
    else:
        default_par['cat_gene_present'] = 0  # chloramphenicol resistance gene absent

    # default initial conditions
    default_init_conds = {}
    for gene in genes:
        default_init_conds['m_' + gene] = 0
        default_init_conds['p_' + gene] = 0
    for misc in miscs:
        default_init_conds[misc] = 0

    # -------- DEFAULT VALUES OF CIRCUIT-SPECIFIC PARAMETERS CAN BE SPECIFIED FROM HERE...
    # toggle switch parameters
    for togswitchnum in (1, 2):  # cycle through toggle switches
        for toggenenum in (1, 2):  # cycle through the genes of the current switch
            default_par['c_tog' + str(togswitchnum) + str(toggenenum)] = 1  # copy no. (nM)
            default_par['a_tog' + str(togswitchnum) + str(toggenenum)] = 10 ** 5 / 2  # promoter strength (unitless)

            # transcription regulation function
            reg_func_string = 'dna(tog' + str(togswitchnum) + str(toggenenum) + '):p_tog' + str(togswitchnum) + str(
                toggenenum % 2 + 1)
            default_par['K_' + reg_func_string] = 2000  # half-saturation constant
            default_par['eta_' + reg_func_string] = 2  # Hill coefficient
            default_par['baseline_tog' + str(togswitchnum) + str(
                toggenenum)] = 0.05  # baseline transcription activation due to leakiness
            default_par['p_tog' + str(togswitchnum) + str(
                toggenenum) + '_ac_frac'] = 1  # active fraction of protein (i.e. share of molecules NOT bound by the inducer)

        # break symmetry for each of the toggle switches
        default_init_conds['m_tog' + str(togswitchnum) + '1'] = 500

    # punisher
    default_par['c_switch'] = 1  # gene concentration (nM)
    default_par['a_switch'] = 150  # promoter strength (unitless)
    default_par['c_int'] = 1  # gene concentration (nM)
    default_par['a_int'] = 1000  # promoter strength (unitless)
    default_par['d_int'] = 6  # integrase protein degradation rate (to avoid unnecessary punishment)
    default_par['c_cat'] = 1  # gene concentration (nM)
    default_par['a_cat'] = 5000  # promoter strength (unitless)

    # transcription regulation function
    default_par['K_switch'] = 100
    default_par['eta_switch'] = 2
    default_par['baseline_switch'] = 0.1
    default_par['p_switch_ac_frac'] = 1  # ACTIVE fraction of protein (i.e. share of molecules bound by the inducer)

    # integrase action
    default_par['K_bI~'] = 100  # dissociation constant for the integrase and DNA (nM)
    default_par['k_sxf'] = 6  # forward strain exchange rate by the integrase(1/h)
    default_par['k_sxr'] = 2.14  # reverse strain exchange rate by the integrase(1/h)
    default_par['k_+syn'] = 0.006  # forward synaptic conformational change rate (1/h)

    # initial conditions for the state of cat gene with respect to the integrase
    default_init_conds['cat_pb'] = 10  # originally, the gene is in functional state (with attP and attB sites)
    default_init_conds['cat_lri1'] = 0  # originally, the gene is in functional state (with attP and attB sites)
    # -------- ...TO HERE

    # default palette and dashes for plotting (5 genes + misc. species max)
    default_palette = ["dodgerblue", "chocolate", "orchid", "crimson",
                       "darkcyan", "darkgoldenrod", "darkviolet", "firebrick"]  # match default palette to genes looping over the five colours we defined
    default_dash = ['solid']
    # match default palette to genes and miscellaneous species, looping over the five colours we defined
    circuit_styles = {'colours': {}, 'dashes': {}}  # initialise dictionary
    for i in range(0, len(genes)):
        circuit_styles['colours'][genes[i]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][genes[i]] = default_dash[i % len(default_dash)]
    for i in range(len(genes), len(genes) + len(miscs)):
        circuit_styles['colours'][miscs[i - len(genes)]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][miscs[i - len(genes)]] = default_dash[i % len(default_dash)]

    # --------  YOU CAN RE-SPECIFY COLOURS FOR PLOTTING FROM HERE...
    # give cat gene states same colours as cat mRNA and protein but different dashes
    circuit_styles['colours']['cat_pb'] = circuit_styles['colours']['cat']
    circuit_styles['colours']['cat_lri1'] = circuit_styles['colours']['cat']

    circuit_styles['dashes']['cat_pb'] = 'solid'  # fully functional gene => solid dash
    circuit_styles['dashes']['cat_lri1'] = 'dashed'  # non-functional => non-solid
    # -------- ...TO HERE

    return default_par, default_init_conds, genes, miscs, name2pos, circuit_styles


# transcription regulation functions
def twotoggles_punisher_F_calc(t ,x, par, name2pos):
    F_cat = 1 # cat is a constitutive gene

    # toggle switch 1
    F_tog11 = par['baseline_tog11'] + (1 - par['baseline_tog11']) * \
        1 / (1 +
             (x[name2pos['p_tog12']] * par['p_tog12_ac_frac'] / par['K_dna(tog11):p_tog12']) ** par['eta_dna(tog11):p_tog12'])  # tog11: regulated by p_tog12

    F_tog12 = par['baseline_tog12'] + (1 - par['baseline_tog12']) * \
                1 / (1 + (x[name2pos['p_tog11']] * par['p_tog11_ac_frac'] / par[
                'K_dna(tog12):p_tog11']) ** par['eta_dna(tog12):p_tog11'])  # tog12: regulated by p_tog11

    # toggle switch 2
    F_tog21 = par['baseline_tog21'] + (1 - par['baseline_tog21']) * \
                1 / (1 + (x[name2pos['p_tog22']] * par['p_tog22_ac_frac'] / par[
                'K_dna(tog21):p_tog22']) ** par['eta_dna(tog21):p_tog22'])  # tog21: regulated by p_tog22
    F_tog22 = par['baseline_tog22'] + (1 - par['baseline_tog22']) * \
                1 / (1 + (x[name2pos['p_tog21']] * par['p_tog21_ac_frac'] / par[
                'K_dna(tog22):p_tog21']) ** par['eta_dna(tog22):p_tog21'])  # tog22: regulated by p_tog21

    # switch and integrase genes regulated by p_switch
    p_switch_dependent_term = (x[name2pos['p_switch']]*par['p_switch_ac_frac']/par['K_switch'])**par['eta_switch']
    F_switch = par['baseline_switch'] + (1 - par['baseline_switch']) * (p_switch_dependent_term/(p_switch_dependent_term+1))
    return jnp.array([F_tog11,
                      F_tog12,
                      F_tog21,
                      F_tog22,
                      F_switch,
                      F_switch,  # integrase co-regulated with the switch gene, hence the same regulation function
                      F_cat])


# ode
def twotoggles_punisher_ode(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state, external inputs
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            par,  # system parameters
            name2pos  # name to position decoder
            ):
    # GET REGULATORY FUNCTION VALUES
    F = F_calc(t, x, par, name2pos)

    # get the Hill equation for integrase-cat DNA binding
    int_cat_hill = (x[name2pos['p_int']]/par['K_bI~'])**4/(1+(x[name2pos['p_int']]/par['K_bI~'])**4)

    # RETURN THE ODE
    return [# mRNAs
            par['func_tog11'] * l * F[name2pos['F_tog11']] * par['c_tog11'] * par['a_tog11'] - (par['b_tog11'] + l) * x[name2pos['m_tog11']],
            par['func_tog12'] * l * F[name2pos['F_tog12']] * par['c_tog12'] * par['a_tog12'] - (par['b_tog12'] + l) * x[name2pos['m_tog12']],
            par['func_tog21'] * l * F[name2pos['F_tog21']] * par['c_tog21'] * par['a_tog21'] - (par['b_tog21'] + l) * x[name2pos['m_tog21']],
            par['func_tog22'] * l * F[name2pos['F_tog22']] * par['c_tog22'] * par['a_tog22'] - (par['b_tog22'] + l) * x[name2pos['m_tog22']],
            par['func_switch'] * l * F[name2pos['F_switch']] * par['c_switch'] * par['a_switch'] - (par['b_switch'] + l) * x[name2pos['m_switch']],
            par['func_int'] * l * F[name2pos['F_int']] * par['c_int'] * par['a_int'] - (par['b_int'] + l) * x[name2pos['m_int']],
            par['func_cat'] * l * F[name2pos['F_cat']] * x[name2pos['cat_pb']] * par['a_cat'] - (par['b_cat'] + l) * x[name2pos['m_cat']],  # NOTE: GENE COPY NO. GIVEN BY CMR_PB!
            # proteins
            (e / par['n_tog11']) * (x[name2pos['m_tog11']] / k_het[name2pos['k_tog11']] / D) * R - (l + par['d_tog11']) * x[name2pos['p_tog11']],
            (e / par['n_tog12']) * (x[name2pos['m_tog12']] / k_het[name2pos['k_tog12']] / D) * R - (l + par['d_tog12']) * x[name2pos['p_tog12']],
            (e / par['n_tog21']) * (x[name2pos['m_tog21']] / k_het[name2pos['k_tog21']] / D) * R - (l + par['d_tog21']) * x[name2pos['p_tog21']],
            (e / par['n_tog22']) * (x[name2pos['m_tog22']] / k_het[name2pos['k_tog22']] / D) * R - (l + par['d_tog22']) * x[name2pos['p_tog22']],
            (e / par['n_switch']) * (x[name2pos['m_switch']] / k_het[name2pos['k_switch']] / D) * R - (l + par['d_switch']) * x[name2pos['p_switch']],
            (e / par['n_int']) * (x[name2pos['m_int']] / k_het[name2pos['k_int']] / D) * R - (l + par['d_int']) * x[name2pos['p_int']],
            (e / par['n_cat']) * (x[name2pos['m_cat']] / k_het[name2pos['k_cat']] / D) * R - (l + par['d_cat']) * x[name2pos['p_cat']],
            # cat gene state with regard to the integrase
            # PB/PBI: fully functional (with attP and attB sites) free or with integrase bound
            -par['k_sxf']*int_cat_hill*x[name2pos['cat_pb']]+par['k_sxr']*x[name2pos['cat_lri1']],
            # LRI1: non-functional (with attL and attR sites) with integrase bound in active conformation - strands synapsed together
            (par['k_sxf'] * int_cat_hill * x[name2pos['cat_pb']] - par['k_sxr'] * x[name2pos['cat_lri1']]) + (
                -par['k_+syn'] * x[name2pos['cat_lri1']] - l*x[name2pos['cat_lri1']]) # no replenishment from LRI2/LR as strands diffuse away
    ]

# TWO TOGGLE SWITCHES WITH SYNTHETIC ADDICTION -------------------------------------------------------------------------
def two_addictive_toggles_initialise():
    # -------- SPECIFY CIRCUIT COMPONENTS FROM HERE...
    genes = ['tog11','tog12','tog21','tog22',  # toggle switches
             'cat11', 'cat12', 'cat21', 'cat22']  # cat genes overlapped with toggle switch genes
    miscs = []  # names of miscellaneous species involved in the circuit
    # -------- ...TO HERE

    # for convenience, one can refer to the species' concs. by names instead of positions in x
    # e.g. x[name2pos['m_xtra']] will return the concentration of mRNA of the gene 'xtra'
    name2pos = {}
    for i in range(0, len(genes)):
        name2pos['m_' + genes[i]] = 8 + i  # mRNA
        name2pos['p_' + genes[i]] = 8 + len(genes) + i  # protein
    for i in range(0, len(miscs)):
        name2pos[miscs[i]] = 8 + len(genes) * 2 + i  # miscellaneous species
    for i in range(0, len(genes)):
        name2pos['k_' + genes[i]] = i  # effective mRNA-ribosome dissociation constants (in k_het, not x!!!)
    for i in range(0, len(genes)):
        name2pos['F_' + genes[i]] = i  # transcription regulation functions (in F, not x!!!)

    # default gene parameters to be imported into the main model's parameter dictionary
    default_par = {}
    for gene in genes:  # gene parameters
        default_par['func_' + gene] = 1.0  # gene functionality - 1 if working, 0 if mutated
        default_par['c_' + gene] = 1.0  # copy no. (nM)
        default_par['a_' + gene] = 100.0  # promoter strength (unitless)
        default_par['b_' + gene] = 6.0  # mRNA decay rate (/h)
        default_par['k+_' + gene] = 60.0  # ribosome binding rate (/h/nM)
        default_par['k-_' + gene] = 60.0  # ribosome unbinding rate (/h)
        default_par['n_' + gene] = 300.0  # protein length (aa)
        default_par['d_' + gene] = 0.0  # rate of active protein degradation - zero by default (/h)

    # chloramphenicol resistance
    if ('cat' in genes):
        default_par['cat_gene_present'] = 1  # chloramphenicol resistance gene present
    else:
        default_par['cat_gene_present'] = 0  # chloramphenicol resistance gene absent
        name2pos['p_cat'] = 0  # placeholder, will never be used but required for correct execution

    # default initial conditions
    default_init_conds = {}
    for gene in genes:
        default_init_conds['m_' + gene] = 0
        default_init_conds['p_' + gene] = 0
    for misc in miscs:
        default_init_conds[misc] = 0

    # -------- DEFAULT VALUES OF CIRCUIT-SPECIFIC PARAMETERS CAN BE SPECIFIED FROM HERE...
    # toggle switch parameters
    for togswitchnum in (1, 2):  # cycle through toggle switches
        for toggenenum in (1, 2):  # cycle through the genes of the current switch
            default_par['c_tog' + str(togswitchnum) + str(toggenenum)] = 1  # copy no. (nM)
            default_par['a_tog' + str(togswitchnum) + str(toggenenum)] = 10 ** 5 / 2  # promoter strength (unitless)

            # transcription regulation function
            reg_func_string = 'dna(tog' + str(togswitchnum) + str(toggenenum) + '):p_tog' + str(togswitchnum) + str(
                toggenenum % 2 + 1)
            default_par['K_' + reg_func_string] = 2000  # half-saturation constant
            default_par['eta_' + reg_func_string] = 2  # Hill coefficient
            default_par['baseline_tog' + str(togswitchnum) + str(
                toggenenum)] = 0.05  # baseline transcription activation due to leakiness
            default_par['p_tog' + str(togswitchnum) + str(
                toggenenum) + '_ac_frac'] = 1  # active fraction of protein (i.e. share of molecules NOT bound by the inducer)

            # overlapped cat gene parameters - same mRNA production rate but different RBS strength
            # !!! therefore par[a_catXY] and par[c_catXY] are DISREGARDED in the ode !!!
            default_par['k+_' + 'cat' + str(togswitchnum) + str(toggenenum)] = default_par['k+_' + 'tog' + str(togswitchnum) + str(toggenenum)] * 0.025

        # break symmetry for each of the toggle switches
        default_init_conds['m_tog' + str(togswitchnum) + '1'] = 500

        # chloramphenicol resistance genes ARE present, just not named 'cat' (so here we amend 'cat_gene_present')
        default_par['cat_gene_present'] = 1
        name2pos['p_cat'] = 8+12  # placeholder, will never be used but required for correct execution

    # -------- ...TO HERE

    # default palette and dashes for plotting (5 genes + misc. species max)
    default_palette = ["dodgerblue", "chocolate", "orchid", "crimson",
                       "teal", "darkgoldenrod", "darkviolet", "firebrick"]  # match default palette to genes looping over the five colours we defined
    default_dash = ['solid']
    # match default palette to genes and miscellaneous species, looping over the five colours we defined
    circuit_styles = {'colours': {}, 'dashes': {}}  # initialise dictionary
    for i in range(0, len(genes)):
        circuit_styles['colours'][genes[i]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][genes[i]] = default_dash[i % len(default_dash)]
    for i in range(len(genes), len(genes) + len(miscs)):
        circuit_styles['colours'][miscs[i - len(genes)]] = default_palette[i % len(default_palette)]
        circuit_styles['dashes'][miscs[i - len(genes)]] = default_dash[i % len(default_dash)]

    # --------  YOU CAN RE-SPECIFY COLOURS FOR PLOTTING FROM HERE...

    # -------- ...TO HERE

    return default_par, default_init_conds, genes, miscs, name2pos, circuit_styles

# transcription regulation function
def two_addictive_toggles_F_calc(t ,x, par, name2pos):
    # toggle switch 1
    F_tog11 = par['baseline_tog11'] + (1 - par['baseline_tog11']) * \
        1 / (1 +
             (x[name2pos['p_tog12']] * par['p_tog12_ac_frac'] / par['K_dna(tog11):p_tog12']) ** par['eta_dna(tog11):p_tog12'])  # tog11: regulated by p_tog12

    F_tog12 = par['baseline_tog12'] + (1 - par['baseline_tog12']) * \
                1 / (1 + (x[name2pos['p_tog11']] * par['p_tog11_ac_frac'] / par[
                'K_dna(tog12):p_tog11']) ** par['eta_dna(tog12):p_tog11'])  # tog12: regulated by p_tog11

    # toggle switch 2
    F_tog21 = par['baseline_tog21'] + (1 - par['baseline_tog21']) * \
                1 / (1 + (x[name2pos['p_tog22']] * par['p_tog22_ac_frac'] / par[
                'K_dna(tog21):p_tog22']) ** par['eta_dna(tog21):p_tog22'])  # tog21: regulated by p_tog22
    F_tog22 = par['baseline_tog22'] + (1 - par['baseline_tog22']) * \
                1 / (1 + (x[name2pos['p_tog21']] * par['p_tog21_ac_frac'] / par[
                'K_dna(tog22):p_tog21']) ** par['eta_dna(tog22):p_tog21'])  # tog22: regulated by p_tog21

    # since the cat genes are co-transcribed with the toggle switch genes, F values are the same
    return jnp.array([F_tog11,
                      F_tog12,
                      F_tog21,
                      F_tog22,
                      # cat genes
                      F_tog11,
                      F_tog12,
                      F_tog21,
                      F_tog22])


# ode
def two_addictive_toggles_ode(F_calc,     # calculating the transcription regulation functions
            t,  x,  # time, cell state, external inputs
            e, l, # translation elongation rate, growth rate
            R, # ribosome count in the cell, resource
            k_het, D, # effective mRNA-ribosome dissociation constants for synthetic genes, resource competition denominator
            par,  # system parameters
            name2pos  # name to position decoder
            ):
    # GET REGULATORY FUNCTION VALUES
    F = F_calc(t, x, par, name2pos)

    # NOTE: overlapping => mRNA conc. ODEs of togXY and catXY are IDENTICAL

    # RETURN THE ODE
    return [# mRNAs
            # toggles
            par['func_tog11'] * l * F[name2pos['F_tog11']] * par['c_tog11'] * par['a_tog11'] - (par['b_tog11'] + l) * x[name2pos['m_tog11']],
            par['func_tog12'] * l * F[name2pos['F_tog12']] * par['c_tog12'] * par['a_tog12'] - (par['b_tog12'] + l) * x[name2pos['m_tog12']],
            par['func_tog21'] * l * F[name2pos['F_tog21']] * par['c_tog21'] * par['a_tog21'] - (par['b_tog21'] + l) * x[name2pos['m_tog21']],
            par['func_tog22'] * l * F[name2pos['F_tog22']] * par['c_tog22'] * par['a_tog22'] - (par['b_tog22'] + l) * x[name2pos['m_tog22']],
            # cats
            par['func_tog11'] * l * F[name2pos['F_tog11']] * par['c_tog11'] * par['a_tog11'] - (par['b_tog11'] + l) * x[name2pos['m_cat11']],
            par['func_tog12'] * l * F[name2pos['F_tog12']] * par['c_tog12'] * par['a_tog12'] - (par['b_tog12'] + l) * x[name2pos['m_cat12']],
            par['func_tog21'] * l * F[name2pos['F_tog21']] * par['c_tog21'] * par['a_tog21'] - (par['b_tog21'] + l) * x[name2pos['m_cat21']],
            par['func_tog22'] * l * F[name2pos['F_tog22']] * par['c_tog22'] * par['a_tog22'] - (par['b_tog22'] + l) * x[ name2pos['m_cat22']],
            # proteins
            # toggles
            (e / par['n_tog11']) * (x[name2pos['m_tog11']] / k_het[name2pos['k_tog11']] / D) * R - (l + par['d_tog11']) * x[name2pos['p_tog11']],
            (e / par['n_tog12']) * (x[name2pos['m_tog12']] / k_het[name2pos['k_tog12']] / D) * R - (l + par['d_tog12']) * x[name2pos['p_tog12']],
            (e / par['n_tog21']) * (x[name2pos['m_tog21']] / k_het[name2pos['k_tog21']] / D) * R - (l + par['d_tog21']) * x[name2pos['p_tog21']],
            (e / par['n_tog22']) * (x[name2pos['m_tog22']] / k_het[name2pos['k_tog22']] / D) * R - (l + par['d_tog22']) * x[name2pos['p_tog22']],
            # cat proteins - p_cat11 is the lumped ODE for ALL cat proteins for easy modelling
            (e / par['n_cat11']) * (x[name2pos['m_tog11']] / k_het[name2pos['k_cat11']] / D) * R - (l + par['d_cat11']) * x[name2pos['p_cat11']] +
            (e / par['n_cat12']) * (x[name2pos['m_tog12']] / k_het[name2pos['k_cat12']] / D) * R - (l + par['d_cat12']) * x[name2pos['p_cat12']] +
            (e / par['n_cat21']) * (x[name2pos['m_tog21']] / k_het[name2pos['k_cat21']] / D) * R - (l + par['d_cat21']) * x[name2pos['p_cat21']] +
            (e / par['n_cat22']) * (x[name2pos['m_tog22']] / k_het[name2pos['k_cat22']] / D) * R - (l + par['d_cat22']) * x[name2pos['p_cat22']],
            0,0,0
    ]