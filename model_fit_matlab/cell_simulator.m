%% cell_simulator.m
% Matlab class enabling simulations of the host cell. 

% The expression of synthetic circuits can be simulated by loading the
% 'heterologous genes' and 'external input' modules (see het_modules and
% ext_inputs folders). Remember to PUSH (obj.push_het()) the associated 
% modules' parameters into the main framework every time you alter them.

%%

classdef cell_simulator
    
    properties (SetAccess = public)
        % VARIABLES USED IN SIMULATIONS
        x0; % initial condition
        t; % time of simulation
        x; % state of the system
        
        % DESCRIBE THE SYSTEM
        init_conditions; % initial conditions according to which x0 is defined when simulation starts
        parameters; % parameters describing the host cell

        % HETEROLOGOUS GENES
        het; % object describing heterologous genes
        num_het=0; % number of heterologous genes
        num_misc=0; % number of miscellaneous species modelled

        % EXTERNAL INPUTS
        ext; % object describing external inputs
        num_ext=0; % number of external inputs
        
        % DEFAULT SETTINGS FOR THE SIMULATOR
        opt = odeset('RelTol', 1e-12, 'AbsTol', 1e-16); % integration tolerances
        tf = 100; % time frame over which we conduct the simulation
        form=cell_formulae; % formulae for rate and activation functions
    end
    
    methods (Access = public)
        %% CONSTRUCTOR
        function obj = cell_simulator(tf)
            % if non-default simulation time suggested, use it
            if nargin == 1
                obj.tf = tf;
            end

            % set default parameters and initial conditions for the host cell
            obj=obj.set_default_parameters();
            obj=obj.set_default_init_conditions();
            
            % set up heterologous genes and external inputs
            obj=obj.load_heterologous_and_external('no_het','no_ext'); % none by default

            % push parameters and initial conditions of het. system into main framework
            obj=obj.push_het();
        end
        

        % SET default parameters (defined in cell_params.m)
        function obj=set_default_parameters(obj)
            obj.parameters=cell_params();
        end
        

        % SET default initial conditions (defined in cell_init_conds.m)
        function obj=set_default_init_conditions(obj)
            obj.init_conditions=cell_init_conds(obj.parameters);
        end
        
        %% HETEROLOGOUS GENE AND EXTERNAL INPUT MODULE MANAGEMENT
        % LOAD heterologous gene and external input modules
        function obj = load_heterologous_and_external(obj,het_sys,ext_sys)
            % LOAD HETEROLOGOUS GENES MODULE
            % access the relevant file
            addpath(genpath([pwd, filesep,'het_modules']));
            het_func=str2func(het_sys);

            % class file describing a synthetic gene system
            obj.het=het_func();

            % get number of heterologous genes
            obj.num_het=size(obj.het.names,2);

            % get number of miscellaneous species
            obj.num_misc=size(obj.het.misc_names,2);

            % LOAD EXTERNAL INPUTS MODULE
            % access the relevant file
            addpath(genpath([pwd, filesep,'ext_inputs']));
            ext_func=str2func(ext_sys);

            % class file describing a synthetic gene system
            obj.ext=ext_func();

            % get number of heterologous genes
            obj.num_ext=size(obj.ext.name,2);

            % push parameters
            obj=obj.push_het();

            % CHECK COMPAITIBILITY
            % is het gene module compatible with external input module?
            if(size(obj.ext.compatible_hets,2)~=0)
                het_in_compatible=false;
                for i=1:size(obj.ext.compatible_hets,2)
                    if strcmp(obj.het.module_name,obj.ext.compatible_hets{i})
                        het_in_compatible=true;
                        break;
                    end
                end
            else
                het_in_compatible=true;
            end

            % does external input module allow the het gene module to work?
            if(size(obj.het.prerequisite_exts,2)~=0)
                ext_in_prerequisite=false;
                for i=1:size(obj.het.prerequisite_exts,2)
                    if strcmp(obj.ext.module_name,obj.het.prerequisite_exts{i})
                        ext_in_prerequisite=true;
                        break;
                    end
                end
            else
                ext_in_prerequisite=true;
            end

            % check compaitibility
            if ~(het_in_compatible && ext_in_prerequisite)
                disp('Incompatible modules! Expect errors')
            end
        end
        

        % PUSH parameters and initial conditions of het. system into main framework
        function obj=push_het(obj)
            obj=obj.push_het_parameters();
            obj=obj.push_het_init_conditions();
        end


        % PUSH heterologous gene parameter values into main framework
        function obj = push_het_parameters(obj)
            % add heterologous genes if there are any
            if(obj.num_het>0)
                for key=keys(obj.het.parameters)
                    obj.parameters(key{1})=obj.het.parameters(key{1});
                end
            end
        end
        

        % PUSH initial conditions for heterologous genes into main framework
        function obj = push_het_init_conditions(obj)
            % add heterologous genes if there are any
            if(obj.num_het>0)
                for key=keys(obj.het.init_conditions)
                    obj.init_conditions(key{1})=obj.het.init_conditions(key{1});
                end
            end
        end
        
        %% SIMULATION
        % CALL the simulator, save the outcome
        function obj = simulate_model(obj)
            obj = obj.set_x0; % set initial condition
            [obj.t, obj.x] = ode15s(@obj.ss_model, [0, obj.tf], [obj.x0], obj.opt);
        end
        

        % DEFINE initial condition according to obj.init_conditions
        function obj = set_x0(obj)
            % NATIVE GENES
            obj.x0 = [
                      % mRNAs;
                      obj.init_conditions('m_a'); % metabolic gene transcripts
                      obj.init_conditions('m_r'); % ribosomal gene transcripts

                      % proteins
                      obj.init_conditions('p_a'); % metabolic proteins
                      obj.init_conditions('R'); % non-inactivated ribosomes

                      % tRNAs
                      obj.init_conditions('tc'); % charged
                      obj.init_conditions('tu'); % uncharged

                      % free ribosomes inactivated by chloramphenicol
                      obj.init_conditions('Bcm');

                      % culture medium's nutrient quality and chloramphenicol concentration
                      obj.init_conditions('s'); % nutrient quality
                      obj.init_conditions('h'); % chloramphenicol levels
                      ];

            % ...ADD HETEROLOGOUS GENES IF THERE ARE ANY
            if(obj.num_het>0)
                x0_het=zeros(2*obj.num_het,1); % initialise
                for i=1:obj.num_het
                    % mRNA
                    x0_het(i)=obj.init_conditions(['m_',obj.het.names{i}]);
                    % protein
                    x0_het(i+obj.num_het)=obj.init_conditions(['p_',obj.het.names{i}]);
                end
                obj.x0=[obj.x0;x0_het]; % concantenate
            end

            % ...ADD MISCELLANEOUS SPECIES IF THERE ARE ANY
            if(obj.num_misc>0)
                x0_misc=zeros(obj.num_misc,1); % initialise
                for i=1:obj.num_misc
                    x0_misc(i)=obj.init_conditions(obj.het.misc_names{i});
                end
                obj.x0=[obj.x0;x0_misc]; % concantenate
            end
        end
        

        % ODEs
        function dxdt = ss_model(obj, t, x)
            % denote obj. parameters as par for convenience
            par = obj.parameters;
            
            % give the state vector entries meaningful names
            m_a = x(1); % metabolic gene mRNA
            m_r = x(2); % ribosomal gene mRNA
            p_a = x(3); % metabolic proteins
            R = x(4); % non-inactivated ribosomes
            tc = x(5); % charged tRNAs
            tu = x(6); % uncharged tRNAs
            Bcm = x(7); % inactivated ribosomes
            s = x(8); % nutrient quality (constant)
            h = x(9); % chloramphenicol concentration
            x_het=x(10:(9+2*obj.num_het+obj.num_misc)); % heterologous genes and miscellaneous synthetic species

            % CALCULATE PHYSIOLOGICAL VARIABLES
            % translation elongation rate
            e=obj.form.e(par,tc);

            % ribosome dissociation constants
            k_a=obj.form.k(e,par('k+_a'),par('k-_a'),par('n_a'));
            k_r=obj.form.k(e,par('k+_r'),par('k-_r'),par('n_r'));

            % heterologous genes rib. dissoc. constants
            k_het=ones(obj.num_het,1); % initialise with default value 1
            if(obj.num_het>0)
                for i=1:obj.num_het
                    k_het(i)=obj.form.k(e,...
                    obj.parameters(['k+_',obj.het.names{i}]),...
                    obj.parameters(['k-_',obj.het.names{i}]),...
                    obj.parameters(['n_',obj.het.names{i}]));
                end
            end

            T=tc./tu... % ratio of charged to uncharged tRNAs 
                .*(1-par('is_fixed_T'))+par('fixed_T').*par('is_fixed_T'); % OR a fixed value (to enable comparison with flux-parity regulation)
            D=(par('K_D')+h)/par('K_D').*...
                (1+(m_a./k_a+m_r./k_r+sum(x_het(1:obj.num_het)./k_het))./...
                (1-par('phi_q'))); % denominator in ribosome competition calculations
            B=R.*(par('K_D')./(par('K_D')+h)-1./D); % actively translating ribosomes (inc. those translating housekeeping genes)

            nu=obj.form.nu(par,tu,s); % tRNA charging rate

            l=obj.form.l(par,e,B); % growth rate

            psi=obj.form.psi(par,T); % tRNA synthesis rate - MUST BE SCALED BY GROWTH RATE

            % GET RNAP ACTIVITY
            rnap_act=l;

            % GET EXTERNAL INPUT
            ext_inp=obj.ext.input(x,t);

            % GET RATE OF EFFECTIVE NUTR. QUAL. CHANGE (for upshifts)
            if(par('is_upshift')==1)
                dsdt = (par('s_postshift') - s) * ...
                    (e./par('n_a')).*(m_a./k_a./D).*R./p_a;
            else
                dsdt = 0;
            end

            % DEFINE DX/DT FOR...
            % ...THE HOST CELL
            dxdt = [
                    % mRNAs
                    rnap_act.*par('c_a').*par('a_a')-(par('b_a')+l).*m_a;
                    rnap_act.*obj.form.F_r(par,T).*par('c_r').*par('a_r')-(par('b_r')+l).*m_r;
                    % ,metabolic protein a
                    (e./par('n_a')).*(m_a./k_a./D).*R-l.*p_a;
                    % ribosomes
                    (e./par('n_r')).*(m_r./k_r./D).*R-l.*R;
                    % tRNAs
                    nu.*p_a-l.*tc-e.*B;
                    psi*rnap_act-l.*tu-nu.*p_a+e.*B;
                    % ribosomes inactivated by chloramphenicol
                    0; % NOW A BLANK: CM ACTION MODELLED DIFFERENTLY
                    % nutrient quality
                    dsdt;
                    % chloramphenicol concentration in the cell (no active degradation)
                    par('diff_h').*(par('h_ext') - h) - l*h;
                    ];

            % ...HETEROLOGOUS GENES
            if(obj.num_het>0)
                dxdt_het=zeros(2*obj.num_het,1); % initialise
                % calculate
                for i=1:obj.num_het
                    % mRNA
                    dxdt_het(i)=rnap_act.*obj.het.regulation(obj.het.names{i},x,ext_inp)...
                        .*par(['c_',obj.het.names{i}]).*par(['a_',obj.het.names{i}])...
                        -(par(['b_',obj.het.names{i}])+l).*x_het(i)...
                        +obj.het.extra_m_term(obj.het.names{i},x,ext_inp);

                    % protein
                    dxdt_het(i+obj.num_het)=(e./par(['n_',obj.het.names{i}])).*(x_het(i)./k_het(i)./D).*R...
                        -l.*x_het(i+obj.num_het)+...
                        obj.het.extra_p_term(obj.het.names{i},x,ext_inp);
                end
                dxdt=[dxdt;dxdt_het]; % concantenate
            end

            % ...MISCELLANEOUS SPECIES
            if(obj.num_misc>0)
                dxdt=[dxdt;obj.het.misc_ode(t,x,ext_inp,l)];
            end
        end
        
    end
end