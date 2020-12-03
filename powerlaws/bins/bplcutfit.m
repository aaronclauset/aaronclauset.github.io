function [alpha_est, lambda_est, L] = bplcutfit(h, boundaries, varargin)

% BPLCUTFIT fits a power-law cutoff model to the data
% Source: http://tuvalu.santafe.edu/~aaronc/powerlaws/bins/
%
%----------
% Options:
%----------
% 1. l = bplcutfit(h, boundaries, 'range', 1.5:0.01:3.5, 0.1:0.01:1)
%    The 'range' option can be specified to restrict search for
%    lambda and beta parameters. In above example, bplcutfit gives 
%    the best looking lambda and beta within the specified ranges. 
%    By default bplcutfit uses matlab's fminsearch function which 
%    in turn uses the Nelder-Mead simplex search algorithm. 
%    Refer following url:
%    (http://www.mathworks.com/help/techdoc/math/bsotu2d.html#bsgpq6p-11)
%
% 2. l = bplcutfit(h, boundaries, 'bmin', 100)
%    The 'bmin' option lets you fix a value such as 100 for bmin. 
%    Note that this value should be one of the values in the 
%    boundaries array. In the above example, 100 cannot be the 
%    last bin boundary. Also, it is advisable to give the fitting
%    procedure atleast two bins to work with. 
%    Note that if 'bmin' value is not one of the elements in 
%    boundaries, bplcutfit chooses the bin boundary which is closest 
%    to the specified value and less than that value. Also, the 
%    Default bmin value is the first bin boundary i.e. 
%    boundaries(1).
%
% Version 1.0 (2012)
% Copyright (C) 2012 Yogesh Virkar (University of Colorado, Boulder)
% Distributed under GNU GPL v3.0
% http://www.gnu.org/copyleft/gpl.html
% BPLCUTFIT comes with ABSOLUTELY NO WARRANTY

rngal = [];
rnglam = [];
bminb = [];

% ---------------------------------------------------------------
% ---------------Parsing command-line arguments------------------
% ---------------------------------------------------------------
i=1;
while i<=length(varargin)
    argok = 1;
    if(ischar(varargin{i}))
        switch varargin{i}
            case 'range', rngal = varargin{i+1}; rnglam = varargin{i+2}; i=i+2;
            case 'bmin', bminb = varargin{i+1}; i=i+1;
            otherwise, argok=0;    
        end
    end
    if ~argok,
        disp(['(BPLCUTFIT) Ignoring invalid argument #' num2str(i+2)]); 
    end
    i=i+1;
end

% ---------------------------------------------------------------
% ------------------------Checking input-------------------------
% ---------------------------------------------------------------

% 1. h must have integer counts.
if isequal(fix(h),h)==0
    fprintf('(BPLCUTFIT) Error: Vector h should be an integer vector.\n');
    return;
end

% 2. h must be non-negative
if ~isempty(find(h<0, 1))
    fprintf('(BPLCUTFIT) Error: Vector h should be non-negative.\n');
    return;
end

% 3. boundaries must have number of elements as one more than 
%    the number in h
if numel(boundaries)~=(numel(h)+1)
    fprintf('(BPLCUTFIT) Error: Incorrect number of elements in either boundaries or h.\n');
    return;
end

% 4. Need atleast 2 bins to work with.
if numel(h)<2
    fprintf('(BPLCUTFIT) Error: I need atleast 2 bins to make this work.\n');
    return;
end

% 5. Checking range vectors
if ~isempty(rngal) && (~isvector(rngal) || min(rngal)<=0 || isempty(rnglam))
    fprintf('(BPLCUTFIT) Error: ''range'' argument must contain a valid vectors; using default.\n');
    rngal = 1.5:0.01:3.5;
    rnglam = 0.1:0.01:1;
end

% 6. Checking bmin option
if ~isempty(bminb) && (~isscalar(bminb) || bminb>=boundaries(end-1))
    fprintf('(BPLCUTFIT) Error: ''bmin'' argument must be a positive value < boundaries(end-1); using default.\n');
    bminb = [];
end

% ---------------------------------------------------------------
% ---------------------------------------------------------------
% ---------------------------------------------------------------

fprintf('Power law with cut off distribution, parameter calculation\n');
fprintf('    Copyright 2012 Yogesh Virkar\n');
fprintf('    Warning: This can be a slow calculation; please be patient.\n');


% Reshape the input vectors
h = reshape(h, numel(h), 1);
boundaries = reshape(boundaries, numel(boundaries), 1);

% Default bmin used is boundaries(1)
if isempty(bminb)
    bminb = boundaries(1);
end

ind = find(boundaries<=bminb, 1, 'last');
bminb = boundaries(ind);
h2 = h(ind:end);
boundaries2  = boundaries(ind:end);

l = boundaries2(1:end-1);
u = boundaries2(2:end);

% Function to be minimized
function fval = plcutoff_mle( vars )
    warning off all;
    
    % alpha should be greater than 1. 
    if(vars(1)<1)
        fval = 10^10;
        return;
    end
    
    a = 1-vars(1);
    x = vars(2)*bminb;
    cnst_C = quadgk(@(t)t.^(a-1).*exp(-t),x,inf,'abstol',1e-12,'reltol', 1e-12, 'MaxIntervalCount', 10000); 
    a = 1-vars(1);
    x = vars(2)*u;
    uEdge = zeros(numel(x), 1);
    for ii=1:numel(x)
        uEdge(ii) = quadgk(@(t)t.^(a-1).*exp(-t),x(ii),inf,'abstol',1e-12,'reltol',1e-12, 'MaxIntervalCount', 10000); 
    end
    a = 1-vars(1);
    x = vars(2)*l;
    lEdge = zeros(numel(x), 1);
    for ii=1:numel(x)
        lEdge(ii) = quadgk(@(t)t.^(a-1).*exp(-t),x(ii),inf,'abstol',1e-12,'reltol',1e-12, 'MaxIntervalCount', 10000); 
    end
    
    temp = h2 .*( log(lEdge-uEdge) - log(cnst_C) );
    fval = -sum(temp);
    
    warning on all; 
end


% Using Grid search 
if ~isempty(rngal) && ~isempty(rnglam)
    fval = zeros(numel(rngal), numel(rnglam));
    for i=1:numel(rngal)
        for j=1:numel(rnglam)
            fval(i,j) = plcutoff_mle([rngal(i) rnglam(j)]);
        end
    end
    [C, I] = min(fval);
    [~, I2] = min(C);
    alpha_est = rngal(I(I2));
    lambda_est = rnglam(I2);
    
% Using fminsearch    
else
    % Note that fminsearch could be sensitive to initial 
    % conditions. A reasonable initial condition could be 
    % startpt = [alpha, 0.0001], where alpha is the estimated alpha
    % value for power-law fit. 
    startpt = [1.5, 0.0001];
    
    vars_est = fminsearch(@plcutoff_mle, startpt);
    alpha_est = vars_est(1);
    lambda_est = vars_est(2);
end

L  = plcutoff_mle([alpha_est lambda_est]);

end