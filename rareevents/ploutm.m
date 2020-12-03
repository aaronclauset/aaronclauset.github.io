function [p, q, s, r, pf]=ploutm(x, varargin)
% PLOUTM estimates the probability of observing a large event size within an
%    empirical data set, where events have categorical covariates.
%
%    Source: http://www.santafe.edu/~aaronc/rareevents/
%
%    Input x must be an n by 2-dimensional vector where the first column
%    gives the events sizes and the second column gives the categorical
%    covariate values (integers) associated with each event. PLOUTM(x) 
%    estimates the probability, under a power-law model of each covariate 
%    distribution's upper tail, of generating at least one value at least 
%    as large as max(x(:,1)), from any of the marginal distributions.
%    
%    The statistical estimation procedure for each marginal distribution 
%    uses the maximum likelihood method of Clauset, Shalizi and Newman 
%    (2009). This procedure estimates a value xmin above which the power-
%    law model is most likely to hold, i.e., Pr(x) \propto x^{-alpha} for 
%    x >= xmin. This approach is combined with a non-parametric bootstrap 
%    of the original data set to estimate the statistical uncertainty of 
%    the model parameters.
%    
%    PLOUTM then returns a vector p that contains the estimated probability
%    and a matrix s that contains the parameters of the ensemble of fitted
%    models.
%
%    PLOUTM automatically detects the type of input data (discrete or
%    continuous) and applies the appropriate version of the power-law 
%    model. For discrete data, if min(x(:,1)) > 1000, PLOUTM uses the 
%    continuous approximation, which is reliable in this regime.
%   
%    The semi-parametric estimation algorithm works as follows:
%    0) First, remove the event max(x) from the data (only if 'cat' is not
%       also invoked)
%    1) For each repetition, bootstrap the data
%    2) Estimate alpha and xmin on these data
%    3) Using the estimated tail model, generate synthetic events drawn
%       from that model
%    4) Count the fraction of these draws that exceed the target size
%
%    Note that this procedure gives no estimate of the validity of the
%    power law for a model of the tail.
%    
%    By default, ploutm draws a single synthetic data set from each set of
%    bootstrap models. If 'draws' is invoked, this number may be increased.
%    The output argument pf then contains the estimated probability for
%    each of the bootstraps that any of the models produces at least one
%    event of the target size. These can be used to construct bootstrap
%    confidence intervals.
%
%    Example:
%       n = 100;
%       x = [(1-rand(n,1)).^(-1/(2.5-1)) 1.*ones(n,1);
%            (1-rand(n,1)).^(-1/(3.0-1)) 2.*ones(n,1)];
%       [p q s] = ploutm(x);              % default behavior
%       [p q s] = ploutm(x,'xmin',2);     % fix xmin=2
%       [p q s] = ploutm(x,'cat',10000);  % set target size
%       [p q s r pf] = ploutm(x,'boots',10000,'draws',100);
%
%    Outputs:
%     p: p(1) is the estimated probability, p(2) is the estimated 
%        standard uncertainty in p(1).
%     q: estimated per-event probability of generating an event at or
%        above the target size.
%     s: a matrix containing parameters for the ensemble of estimated
%        models, of the form [alpha xmin ntail rho], which can be used for
%        visualizaton and other purposes.
%     r: vector of p(:) for each covariate type
%    
%    For more information, try 'type ploutm'
%
%    See also PLFIT and PLOUT

% Version 1.0    (2011 December)
% Version 1.0.1  (2012 March)
% Copyright (C) 2011-2012 Aaron Clauset (Univerity of Colorado, Boulder)
% Distributed under GPL 2.0
% http://www.gnu.org/copyleft/gpl.html
% PLOUTM comes with ABSOLUTELY NO WARRANTY
% 
% Notes:
% 
% 1. To explicitly specify a target event size
%    
%       p = ploutm(x,'cat',10000);
%    
% 2. In order to implement the integer-based methods in Matlab, the numeric
%    maximization of the log-likelihood function is used. This requires
%    that we specify the range of scaling parameters considered. We set
%    this range to be [1.50 : 0.01 : 3.50] by default. This vector can be
%    set by the user like so,
%    
%       p = ploutm(x,'range',[1.001:0.001:5.001]);
%    
% 3. PLOUTM can be told to limit the range of values considered by PLFIT as
%    estimates for xmin in two ways. First, it can be instructed to sample 
%    these possible values like so,
%    
%       p = ploutm(x,'sample',100);
%    
%    which uses 100 uniformly distributed values on the sorted list of
%    unique values in the data set. Second, it can simply omit all
%    candidates above a hard limit, like so
%    
%       p = ploutm(x,'limit',3.4);
%
%    In each case, the corresponding arguments are passed through to PLFIT.
%    
% 4. A fixed xmin value may be specified like so
%    
%       p = ploutm(x,'xmin',10);
%    
%    In the case of discrete data, it rounds the argument to the nearest
%    integer. In this case, PLFIT is not called and alpha is estimated
%    conditioned on the specified xmin value. Otherwise, the algorithm
%    proceeds as before.
% 
% 5. Text output to stdout can be silenced
%    
%       p = ploutm(x,'silent');
%    

range   = [];
sample  = [];
limit   = [];
xminx   = [];
xcat    = [];
silent  = false;
nowarn  = false;
f_xmin  = 'FREE';      % (default) 
f_cat   = 'DEFAULT';   % (default) 
f_Na    = 'DEFAULT';   % (default) 
f_Nb    = 'DEFAULT';   % (default) 
persistent rand_state;

% parse command-line parameters; trap for bad input
i=1; 
while i<=length(varargin), 
  argok = 1; 
  if ischar(varargin{i}), 
    switch varargin{i},
        case 'range',        range   = varargin{i+1}; i = i + 1;
        case 'sample',       sample  = varargin{i+1}; i = i + 1;
        case 'limit',        limit   = varargin{i+1}; i = i + 1;
        case 'cat',          xcat    = varargin{i+1}; i = i + 1; f_cat  = 'USER';
        case 'xmin',         xminx   = varargin{i+1}; i = i + 1; f_xmin = 'FIXED';
        case 'boots',        xNa     = varargin{i+1}; i = i + 1; f_Na   = 'USER';
        case 'draws',        xNb     = varargin{i+1}; i = i + 1; f_Nb   = 'USER';
        case 'silent',       silent  = true;
        case 'nowarn',       nowarn  = true;
        otherwise, argok=0; 
    end
  end
  if ~argok, 
    disp(['(PLOUTM) Ignoring invalid argument #' num2str(i+1)]); 
  end
  i = i+1; 
end

% 1a. -- grab size data
y = x(:,1);

% 1b. -- check input arguments
if ~isempty(range) && (~isvector(range) || min(range)<=1),
	fprintf('(PLOUTM) Error: ''range'' argument must contain a vector; using default.\n');
    range = [];
end;
if ~isempty(sample) && (~isscalar(sample) || sample<2),
	fprintf('(PLOUTM) Error: ''sample'' argument must be a positive integer > 1; using default.\n');
    sample = [];
end;
if ~isempty(limit) && (~isscalar(limit) || limit<min(y)),
	fprintf('(PLOUTM) Error: ''limit'' argument must be a positive value >= 1; using default.\n');
    limit = [];
end;
if ~isempty(xminx) && (~isscalar(xminx) || xminx>=max(y)),
	fprintf('(PLOUTM) Error: ''xmin'' argument must be a positive value < max(x); using default behavior.\n');
    xminx = [];
end;
if strcmp(f_cat,'USER') && (~isscalar(xcat) || xcat<=min(y)),
	fprintf('(PLOUTM) Error: ''cat'' argument must be a positive value > min(x); using default behavior.\n');
    xminx = [];
end;

% 1c. -- build argument string for PLFIT
varargin = {};
if     ~isempty(range),  varargin = {'range',range};
elseif ~isempty(sample), varargin = {'sample',sample};
elseif ~isempty(limit),  varargin = {'limit',limit};
end;
if silent || nowarn, varargin = [varargin {'nowarn'}]; end;

% 1d. -- select discrete or continuous valued power-law model
if     isempty(setdiff(y,floor(y))), f_dattype = 'INTS';
elseif isreal(y),    f_dattype = 'REAL';
end;
if strcmp(f_dattype,'INTS') && (min(y) > 1000 && length(y)>100),
    f_dattype = 'REAL';
end;

% 1e. -- fix xmin value, if necessary
if strcmp(f_xmin,'FIXED')
    switch f_dattype
        case 'INTS', xmin = round(xminx);
        case 'REAL', xmin = xminx;
    end;
end;

% 1f. -- choose target event size
switch f_cat
    case 'DEFAULT',  cat   = max(y);          % cat = largest observed value
                     catty = x(x(:,1)==cat,2);% get cat type
                     x(x(:,1)==cat,:) = [];   % remove this value from the data
    case 'USER',     cat = xcat;     % cat = user specified
end;

% 1g. -- set number of Monte Carlo repetitions
switch f_Na
    case 'DEFAULT',  Na = 1000;  % (default) 10^3 bootstraps
    case 'USER',     Na = xNa;   % 
end;
switch f_Nb
    case 'DEFAULT',  Nb = 1;     % (default) 1 sample per bootstrap model
    case 'USER',     Nb = xNb;   %
end;

% 1h. -- initialize random number generator
if isempty(rand_state)
    rand_state = cputime;
    rand('twister',sum(100*clock)); % deprecated
end;

% 2. -- do the estimation
c       = unique(x(:,2)); % unique covariate values
u       = length(c);      % number of unique covariate values
ncat    = zeros(Na,u);    % intermediate results: #cat in each covariate
acat    = zeros(Na,1);    % intermediate results: prob >=1 cat in any covariate
results = zeros(u*Na,6);  % intermediate results: [ type alpha xmin ntail rho ncat]
n       = size(x,1);      % number of observations
ij      = 1;

if ~silent,
    fprintf('\nPower-law distribution, outlier calculation (with covariates)\n');
    fprintf('   Copyright 2011-2012 Aaron Clauset\n');
    fprintf('   Warning: This can be a slow calculation; please be patient.\n');
    switch f_dattype
        case 'INTS', fprintf('   n      = %i\n   boots  = %i (%i draws)\n   target = %i\n',n,Na,Nb,cat);
        case 'REAL', fprintf('   n      = %i\n   boots  = %i (%i draws)\n   target = %6.4f\n',n,Na,Nb,cat);
    end;
end;

if ~silent, tic; end;
for i=1:Na
    % 1a. bootstrap full empirical data set
    g = x(ceil(n.*rand(n,1)),:);
    
    % 1b. for each covariate type
    rho  = zeros(Nb,2*u);
    parm = zeros(u,3);
    for ell=1:u
        z  = g(g(:,2)==c(ell),1); % get severities
        
        % 2. estimate alpha (and possibly xmin)
        switch f_xmin
            case 'FREE'
                % 2a. estimate alpha,xmin via PLFIT (chooses discrete vs. continuous as above)
                [alpha,xmin] = plfit(z,varargin{:});
                ntail = sum(z>=xmin);
            case 'FIXED'
                % 2b. estimate alpha alone
                ztail = z(z>=xmin);
                ntail = length(ztail);
                switch f_dattype
                    case 'INTS'
                        if isempty(range)
                            % (default) non-parametric optimization (can be slow)
                            L = @(a) -(-a.*sum(log(ztail)) - ntail.*log(zeta(a) - sum( (1:xmin-1).^(-a) )));
                            alpha = fminsearch(L,2.5);
                        else
                            % grid search
                            vec  = range;
                            zvec = zeta(vec);
                            L  = -Inf*ones(size(vec));
                            for k=1:length(vec)
                                L(k) = -vec(k)*sum(log(ztail)) - ntail*log(zvec(k) - sum((1:xmin-1).^-vec(k)));
                            end
                            [~,I] = max(L);
                            alpha = vec(I);
                        end;
                    case 'REAL'
                        alpha = 1 + ntail ./ sum( log(ztail./xmin) );
                end;
        end;
        % 3. set threshold for catastrophic event
        switch f_dattype
            case 'INTS'
            thresh = sum( (((xmin:cat).^-alpha))'./ (zeta(alpha) - sum((1:xmin-1).^-alpha)) );
            case 'REAL'
            thresh = 1-(cat./xmin).^(1-alpha);
        end;
        % 4. estimate probability under fitted model
        if strcmpi(f_cat,'DEFAULT') && c(ell)==catty, ntail = 1+ntail; end;
        %rho = zeros(Nb,2);
        for j=1:Nb
            r = rand(ntail,1); % draw ntail from U(0,1)
            rho(j,2*ell-1:2*ell) = [any(r >= thresh) sum(r >= thresh)];
        end;
        % 5. record intermediate results
        parm(ell,:) = [alpha xmin ntail];
        ncat(i,ell) = mean(rho(:,2*ell)); % mean #events >= cat
    end;
    % 6. record final results
    for ell=1:u
        % [ type | alpha | xmin | ntail | prob >=1 cat | # >= cat ]
        results(ij,:) = [c(ell) parm(ell,1) parm(ell,2) parm(ell,3) mean(rho(:,2*ell-1)) ncat(i,ell)];
        ij=ij+1;
    end;
    acat(i) = mean(any(rho(:,1:2:end)'));
    % 6. track progress; estimate completion time
    if ~silent && Na>=100 && mod(i,Na/100)==0
        fprintf('[%3.0f%% done (%i boots)]    p-hat = %5.3f +/- %5.3f    (%4.2fm <-/-> %4.2fm)\n',i*100/Na,i,mean(acat(1:i)),std(acat(1:i))./sqrt(i),toc/60,(Na/i-1)*(toc/60));
    end;
end;
% 5. -- report the results
% p  = [ mean(Pr >= 1 cat in any covariate) std(Pr)./sqrt(Na) ]
p  = [mean(acat) std(acat)./sqrt(Na)];
% pf = distribution of p
pf = acat;
% q = Pr(X>=x | x>=xmin) * Pr(X >= xmin)
%   = mean( (#cat events)./(ntail) )
q = mean(sum(ncat,2)./n);
% s = [ alpha xmin ntail rho type ]
% format chosen to ensure compatibility with pleplot
s = results(:,[2:5 1]);
% r = [ type mean(rho | type) std(rho | type)./sqrt(Na) ]
r = zeros(u,3);
for ell=1:u
    grab     = results(results(:,1)==c(ell),[5 6]);
    r(ell,:) = [c(ell) mean(grab(:,1)) std(grab(:,1))./sqrt(Na)];
end;
if ~silent
    fprintf('p-hat = %5.3f +/- %5.3f\n',p(1),p(2));
    fprintf('q-hat = %15.13f\n',q);
end;



