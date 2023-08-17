%% update_a_a.m
% For the vector of fitted parameter values, find the more accurate value
% of a_a as described in the Supplementary Information Section 2.2

%% CLEAR

clear
close all

%% DEFINE the reference growth rate and total mRNA synthesis rate in the cell

ref_l = log(2); % 1 doubling/h
ref_A = 6.12e7; % 6.12*10^7 nucleotides/h = 1.02*10^5 nucl/min - Bremer and Dennis 2008, EcoSal Plus

%% SPECIFY  the fitted parameter vector

theta=[0.682143, 4405.67, 2095.55];

%% SET UP the simulator

sim=cell_simulator; % initialise simulator

% parameters for getting steady state
sim.tf = 12; % single integraton step timeframe
Delta = 0; % threshold that determines if we're in steady state
Max_iter = 4; % maximum no. iterations (checking if SS reached over first 750 h)

sim.opt = odeset('reltol',1.e-6,'abstol',1.e-9); % more lenient integration tolerances for speed

%% SPECIFY range of nutrient qualit for which we run the simulation

nutr_quals=linspace(0.01,1,1000);

%% INITIALISE the arrays where model predictions will be stored

l_map=zeros(size(nutr_quals));
phia_map=zeros(size(nutr_quals));
Fr_map=zeros(size(nutr_quals));

%% GET model predictions

for j=1:size(nutr_quals,2)
    disp(nutr_quals(j))

    % reset simulator
    sim.parameters=cell_params();
    
    % TRIAL
    sim.parameters('a_a') = 3.89e5; % metabolic prot. transcription rate (/h)
    sim.parameters('a_r') = sim.parameters('a_a').*theta(1); % ribosome transcription rate (/h) - rescaled!
    sim.parameters('nu_max') = theta(2); % max metabolic rate (/h)
    sim.parameters('K_e') = theta(3); % elongation rate Hill constant (nM)
    sim.parameters('K_nut') = theta(3); % tRNA charging rate Hill constant (nM)

    sim.init_conditions=cell_init_conds(sim.parameters); % reset intial conditions
    
    % set nutrient quality
    sim.init_conditions('s')=nutr_quals(j);
    sim=sim.push_het();

    % Run
    sim.parameters('is_fixed_F_r')=0; % F_r regulated now
    ss = get_steady(sim,Delta,Max_iter);
    [l, phi_a, F_r] = get_lphiaFr(sim,ss);
    l_map(j)=l;
    phia_map(j)=phi_a;
    Fr_map(j)=F_r;
end

%% FIND the reference ppGpp concentration (for l=1/h)
distance_to_ref=100; % initialise the distance to reference growth rate with unreasonably high number
for i=1:size(l_map,2)
    if(abs(l_map(i)-ref_l)<abs(distance_to_ref)) % if the current distance is the smallest so far, this is the neww reference
        distance_to_ref=l_map(i)-ref_l;
        closest_l=l_map(i);
        closest_phia=phia_map(i);
        closest_Fr=Fr_map(i);
        closest_nutrqual=nutr_quals(i);
    end
end

%% PRINT obtained values

disp(['Reference growth rate = ', num2str(ref_l), ' h^(-1)'])
disp(['Sim. growth rate closest to ref. = ', num2str(closest_l), ' h^(-1)'])
disp(['Corresponding nutrient quality = ', num2str(closest_nutrqual)])
disp(['Metab. prot. mass frac. at this growth rate = ',num2str(closest_phia)])
disp(['F_r value at this growth rate = ', num2str(closest_Fr)])

%% CALCULATE the updated estimate for a_a

a_a_upd = ref_A / ref_l / ...
    (75*sim.parameters('phi_q')/closest_phia + ...
    75 * closest_Fr * theta(1) + ...
    75);
disp(['Updated a_a value = ', num2str(a_a_upd)])

%% CALCULATE the updated estimate for a_r
a_r_upd = a_a_upd * theta(1);
disp(['Updated a_r value = ',num2str(a_r_upd)])

%% FUNCTION for getting growth rate, translation elongation rate and rib. mass fraction from the system's steady state

function [l, phi_a, F_r]=get_lphiaFr(sim,ss)
    % get growth rate and ribosome mass fraction
    par=sim.parameters;
    m_a = ss(1);
    m_r = ss(2);
    p_a = ss(3);
    R = ss(4);
    tc = ss(5);
    tu = ss(6);
    Bcm = ss(7);
    s = ss(8);
    h = ss(9);
    ss_het=ss(10 : (9+2*sim.num_het) ).';

    phi_a=p_a.*par('n_a')./par('M');% mass fraction of metabolic proteins!

    e=sim.form.e(par,tc); % translation elongation rate

    % ribosome dissociation constants
    k_a=sim.form.k(e,par('k+_a'),par('k-_a'),par('n_a'));
    k_r=sim.form.k(e,par('k+_r'),par('k-_r'),par('n_r'));
    k_het=ones(sim.num_het,1);

    % ribosome dissociation constants for heterologous genes
    if(sim.num_het>0)
        for j=1:sim.num_het
            k_het(j)=sim.form.k(e,...
            sim.parameters(['k+_',sim.het.names{j}]),...
            sim.parameters(['k-_',sim.het.names{j}]),...
            sim.parameters(['n_',sim.het.names{j}]));
        end
    end

    D=(par('K_D')+h)/par('K_D').*...
        (1+(m_a./k_a+m_r./k_r+sum(ss_het(1:sim.num_het)./k_het))./...
        (1-par('phi_q'))); % denominator in ribosome competition calculations
    B=R.*(par('K_D')./(par('K_D')+h)-1./D); % actively translating ribosomes (inc. those translating housekeeping genes)

    l=sim.form.l(par,e,B); % growth rate!
    
    T=tc./tu; % inverse of ppGpp level
    F_r = sim.form.F_r(par,T); % ribosomal gene and tRNA transcription reg. func. !
end