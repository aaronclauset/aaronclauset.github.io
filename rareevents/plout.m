function [p, q, s]=plout(x, varargin)
% PLOUT estimates the probability of observing a large event size within an
%    empirical data set.
%
%    Source: http://www.santafe.edu/~aaronc/rareevents/
%
%    Given a 1-dimensional vector of event sizes x, PLOUT(x) estimates the
%    probability, under a power-law model of the distribution's upper tail,
%    of generating at least one value at least as large as max(x).
%    
%    The statistical estimation procedure is based on the maximum
%    likelihood method of Clauset, Shalizi and Newman (2009). This
%    procedure estimates a value xmin above which the power-law model is
%    most likely to hold, i.e., Pr(x) \propto x^{-alpha} for x >= xmin.
%    This approach is combined with a non-parametric bootstrap to estimate 
%    the statistical uncertainty of the model parameters.
%    
%    PLOUT then returns a vector p that contains the estimated probability
%    and a matrix s that contains the parameters of the ensemble of fitted
%    models.
%
%    PLOUT automatically detects the type of input data (discrete or
%    continuous) and applies the appropriate version of the power-law 
%    model. For discrete data, if min(x) > 1000, PLOUT uses the continuous
%    approximation, which is reliable in this regime.
%   
%    The semi-parametric estimation algorithm works as follows:
%    0) First, remove the event max(x) from the data
%    1) For each repetition, bootstrap the remaining data
%    2) Estimate alpha and xmin on these data
%    3) Using the estimated tail model, generate synthetic events drawn
%       from that model
%    4) Count the fraction of these draws that exceed the target size
%
%    Note that this procedure gives no estimate of the validity of the
%    power law for a model of the tail.
%
%    By default, plout draws a single synthetic data set from each set of
%    bootstrap models. If 'draws' is invoked, this number may be increased.
%    The output argument s(:,4) then contains the estimated probability for
%    each of the bootstraps of producing at least one event of the target
%    size under that model. These can be used to construct bootstrap
%    confidence intervals.
%    
%    Example:
%       x = (1-rand(100,1)).^(-1/(2.5-1));
%       [p q s] = plout(x);              % default behavior
%       [p q s] = plout(x,'xmin',2);     % fix xmin=2
%       [p q s] = plout(x,'cat',10000);  % set target size
%       [p q s] = plout(x,'boots',10000,'draws',100);
%
%    Outputs:
%     p: p(1) is the estimated probability, p(2) is the estimated 
%        standard uncertainty in p(1).
%     q: estimated per-event probability of generating an event at or
%        above the target size.
%     s: a matrix containing parameters for the ensemble of estimated
%        models, of the form [alpha xmin ntail rho], which can be used for
%        visualizaton and other purposes.
%    
%    For more information, try 'type plout'
%
%    See also PLFIT and PLOUTM

% Version 1.0    (2011 December)
% Version 1.0.1  (2012 March)
% Copyright (C) 2011-2012 Aaron Clauset (Univerity of Colorado, Boulder)
% Distributed under GPL 2.0
% http://www.gnu.org/copyleft/gpl.html
% PLOUT comes with ABSOLUTELY NO WARRANTY
% 
% Notes:
% 
% 1. To explicitly specify a target event size
%    
%       p = plout(x,'cat',10000);
%    
% 2. In order to implement the integer-based methods in Matlab, the numeric
%    maximization of the log-likelihood function is used. This requires
%    that we specify the range of scaling parameters considered. We set
%    this range to be [1.50 : 0.01 : 3.50] by default. This vector can be
%    set by the user like so,
%    
%       p = plout(x,'range',[1.001:0.001:5.001]);
%    
% 3. PLOUT can be told to limit the range of values considered by PLFIT as
%    estimates for xmin in two ways. First, it can be instructed to sample 
%    these possible values like so,
%    
%       p = plout(x,'sample',100);
%    
%    which uses 100 uniformly distributed values on the sorted list of
%    unique values in the data set. Second, it can simply omit all
%    candidates above a hard limit, like so
%    
%       p = plout(x,'limit',3.4);
%
%    In each case, the corresponding arguments are passed through to PLFIT.
%    
% 4. A fixed xmin value may be specified like so
%    
%       p = plout(x,'xmin',10);
%    
%    In the case of discrete data, it rounds the argument to the nearest
%    integer. In this case, PLFIT is not called and alpha is estimated
%    conditioned on the specified xmin value. Otherwise, the algorithm
%    proceeds as before.
% 
% 5. Text output to stdout can be silenced
%    
%       p = plout(x,'silent');
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
    disp(['(PLOUT) Ignoring invalid argument #' num2str(i+1)]); 
  end
  i = i+1; 
end

% 1a. -- reshape input into a 1-dimensional vector
x = reshape(x,numel(x),1);

% 1b. -- check input arguments
if ~isempty(range) && (~isvector(range) || min(range)<=1),
	fprintf('(PLOUT) Error: ''range'' argument must contain a vector; using default.\n');
    range = [];
end;
if ~isempty(sample) && (~isscalar(sample) || sample<2),
	fprintf('(PLOUT) Error: ''sample'' argument must be a positive integer > 1; using default.\n');
    sample = [];
end;
if ~isempty(limit) && (~isscalar(limit) || limit<min(x)),
	fprintf('(PLOUT) Error: ''limit'' argument must be a positive value >= 1; using default.\n');
    limit = [];
end;
if ~isempty(xminx) && (~isscalar(xminx) || xminx>=max(x)),
	fprintf('(PLOUT) Error: ''xmin'' argument must be a positive value < max(x); using default behavior.\n');
    xminx = [];
end;
if strcmp(f_cat,'USER') && (~isscalar(xcat) || xcat<=min(x)),
	fprintf('(PLOUT) Error: ''cat'' argument must be a positive value > min(x); using default behavior.\n');
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
if     isempty(setdiff(x,floor(x))), f_dattype = 'INTS';
elseif isreal(x),    f_dattype = 'REAL';
end;
if strcmp(f_dattype,'INTS') && (min(x) > 1000 && length(x)>100),
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
    case 'DEFAULT',  cat = max(x);   % cat = largest observed value
                     x(x==cat) = []; % remove this value from the data
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
results = zeros(Na,5);  % intermediate results: [ alpha xmin ntail rho ncat]
n       = length(x);  % number of observations in original data

if ~silent,
    fprintf('\nPower-law distribution, outlier calculation\n');
    fprintf('   Copyright 2011-2012 Aaron Clauset\n');
    fprintf('   Warning: This can be a slow calculation; please be patient.\n');
    switch f_dattype
        case 'INTS', fprintf('   n      = %i\n   boots  = %i (%i draws)\n   target = %i\n',n,Na,Nb,cat);
        case 'REAL', fprintf('   n      = %i\n   boots  = %i (%i draws)\n   target = %6.4f\n',n,Na,Nb,cat);
    end;
end;

if ~silent, tic; end;
for i=1:Na
    % 1. bootstrap of empirical data
    z = x(ceil(n.*rand(n,1)));
    
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
    rho = zeros(Nb,2);
    if strcmpi(f_cat,'DEFAULT'), ntail = 1+ntail; end;
    for j=1:Nb
        r        = rand(ntail,1); % draw 1+ntail from U(0,1)
        rho(j,:) = [any(r >= thresh) sum(r >= thresh)];
    end;
    % 5. record results
    results(i,:) = [alpha xmin ntail mean(rho(:,1)) mean(rho(:,2))];
    % 6. track progress; estimate completion time
    if ~silent && Na>=100 && mod(i,Na/100)==0
        fprintf('[%3.0f%% done (%i boots)]    p-hat = %5.3f +/- %5.3f    (%4.2fm <-/-> %4.2fm)\n',i*100/Na,i,mean(results(1:i,4)),std(results(1:i,4))./sqrt(i),toc/60,(Na/i-1)*(toc/60));
    end;
end;
% 5. -- report the results
% p = [ mean(rho) std(rho)./sqrt(Na) ]
p = [mean(results(:,4)) std(results(:,4))./sqrt(Na)];
% q = Pr(X>=x | x>=xmin) * Pr(X >= xmin)
%   = mean( (#cat events)./(ntail) )
q = mean(results(:,5)./n);
% s = [ alpha xmin ntail rho ]
s = results(:,1:4);
if ~silent
    fprintf('p-hat = %5.3f +/- %5.3f\n',p(1),p(2));
    fprintf('q-hat = %15.13f\n',q);
end;

