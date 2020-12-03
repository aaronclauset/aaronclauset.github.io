function [lambda_est, beta_est, L] = bstexpfit(h, boundaries, varargin)

% BSTEXPFIT fits a stretched exponential model to the data
% Source: http://tuvalu.santafe.edu/~aaronc/powerlaws/bins/
%
%----------
% Options:
%----------
% 1. l = bstexpfit(h, boundaries, 'range', 0.1:0.01:1, 1:0.01:2)
%    The 'range' option can be specified to restrict search for
%    lambda and beta parameters. In above example, bstexpfit gives 
%    the best looking lambda and beta within the specified ranges. 
%    By default bstexpfit uses matlab's fminsearch function which 
%    in turn uses the Nelder-Mead simplex search algorithm. 
%    Refer following url:
%    (http://www.mathworks.com/help/techdoc/math/bsotu2d.html#bsgpq6p-11)
%
% 2. l = bstexpfit(h, boundaries, 'bmin', 100)
%    The 'bmin' option lets you fix a value such as 100 for bmin. 
%    Note that this value should be one of the values in the 
%    boundaries array. In the above example, 100 cannot be the 
%    last bin boundary. Also, it is advisable to give the fitting
%    procedure atleast two bins to work with. 
%    Note that if 'bmin' value is not one of the elements in 
%    boundaries, bstexpfit chooses the bin boundary which is closest 
%    to the specified value and less than that value. Also, the 
%    Default bmin value is the first bin boundary i.e. 
%    boundaries(1).
%
% Version 1.0 (2012)
% Copyright (C) 2012 Yogesh Virkar (University of Colorado, Boulder)
% Distributed under GNU GPL v3.0
% http://www.gnu.org/copyleft/gpl.html
% BSTEXPFIT comes with ABSOLUTELY NO WARRANTY

rnglam = [];
rngbet = [];
bminb = [];

% ---------------------------------------------------------------
% ---------------Parsing command-line arguments------------------
% ---------------------------------------------------------------
i=1;
while i<=length(varargin)
    argok = 1;
    if(ischar(varargin{i}))
        switch varargin{i}
            case 'range', rnglam = varargin{i+1}; rngbet = varargin{i+2}; i=i+2;
            case 'bmin', bminb = varargin{i+1}; i=i+1;
            otherwise, argok=0;    
        end
    end
    if ~argok,
        disp(['(BSTEXPFIT) Ignoring invalid argument #' num2str(i+2)]); 
    end
    i=i+1;
end

% ---------------------------------------------------------------
% ------------------------Checking input-------------------------
% ---------------------------------------------------------------

% 1. h must have integer counts.
if isequal(fix(h),h)==0
    fprintf('(BSTEXPFIT) Error: Vector h should be an integer vector.\n');
    return;
end

% 2. h must be non-negative
if ~isempty(find(h<0, 1))
    fprintf('(BSTEXPFIT) Error: Vector h should be non-negative.\n');
    return;
end

% 3. boundaries must have number of elements as one more than 
%    the number in h
if numel(boundaries)~=(numel(h)+1)
    fprintf('(BSTEXPFIT) Error: Incorrect number of elements in either boundaries or h.\n');
    return;
end

% 4. Need atleast 2 bins to work with.
if numel(h)<2
    fprintf('(BSTEXPFIT) Error: I need atleast 2 bins to make this work.\n');
    return;
end

% 5. Checking range vectors
if ~isempty(rnglam) && (~isvector(rnglam) || min(rnglam)<=0 || isempty(rngbet))
    fprintf('(BSTEXPFIT) Error: ''range'' argument must contain a valid vectors; using default.\n');
    rnglam = 0.1:0.01:1;
    rngbet = 1:0.01:2;
end

% 6. Checking bmin option
if ~isempty(bminb) && (~isscalar(bminb) || bminb>=boundaries(end-1))
    fprintf('(BSTEXPFIT) Error: ''bmin'' argument must be a positive value < boundaries(end-1); using default.\n');
    bminb = [];
end

% ---------------------------------------------------------------
% ---------------------------------------------------------------
% ---------------------------------------------------------------

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


if ~isempty(rnglam) && ~isempty(rngbet)
    rnglam = reshape(rnglam, 1, numel(rnglam));
    rngbet = reshape(rngbet, 1, numel(rngbet));
    LAMBDA = repmat(repmat(rnglam, numel(l), 1), [1, 1, numel(rngbet)]);
    temp = reshape(rngbet, [1 1 numel(rngbet)]);
    BETA = repmat(temp, numel(l), numel(rnglam));
    L = repmat(repmat(l, 1, numel(rnglam)), [1, 1, numel(rngbet)]);
    U = repmat(repmat(u, 1, numel(rnglam)), [1, 1, numel(rngbet)]);
    H = repmat(repmat(h2, 1, numel(rnglam)), [1, 1, numel(rngbet)]);
    
    temp = H.*(log(exp(-LAMBDA.*(L.^BETA)) - exp(-LAMBDA.*(U.^BETA))) + LAMBDA.*(bminb.^BETA));
    
    fval = reshape(-sum(temp,1), numel(rnglam), numel(rngbet)); clear ('temp', 'L', 'U', 'H', 'BETA', 'LAMBDA');
    
    [C, I] = min(fval);
    [~, I2] = min(C);
    lambda_est = rnglam(I(I2));
    beta_est = rngbet(I2);
    
else
    hnd = @(vars) -sum(h2.*(log(exp(-vars(1).*(l).^vars(2)) - exp(-vars(1).*(u.^vars(2)))) + (vars(1)*bminb^vars(2))));
    mymaxfun = 10000;
    mymaxiter = 10000;
    options = optimset('MaxFunEvals', mymaxfun, 'MaxIter', mymaxiter);
    vars_estimate = fminsearch(hnd, [0.1, 0.1], options);
    lambda_est = vars_estimate(1);
    beta_est = vars_estimate(2);
end


L = -sum(h2.*(log(exp(-lambda_est.*(l).^beta_est) - exp(-lambda_est.*(u.^beta_est))) + (lambda_est*bminb^beta_est)));

end