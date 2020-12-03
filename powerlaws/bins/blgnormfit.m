function [mu_est, sigma_est, L] = blgnormfit(h, boundaries, murng, sigrng, varargin)

% BLGNORMFIT fits a log-normal model to the data 
% Source: http://tuvalu.santafe.edu/~aaronc/powerlaws/bins/
%
% Note that unlike the other fitting procedures, blgnormfit uses
% only grid search and does not default to fminsearch. Hence, it is
% required to specify the range for mu and sigma (denoted by murng 
% and sigrng) while calling blgnormfit. 
%
% The default options for murng and sigrng can be specified like this,
% [m, s, L] = blgnormfit(h, boundaries, [], []);
% The default ranges used are: 
% murng = -50:1:50 and sigrng = 1:0.1:20;
%
% The other options are listed below:
%
%----------
% Options:
%----------
%
% 1. [mu, sig] = blgnormfit(h, boundaries, murng, sigrng, 'bmin', 100)
%    The 'bmin' option lets you fix a value such as 100 for bmin. 
%    Note that this value should be one of the values in the 
%    boundaries array. In the above example, 100 cannot be the 
%    last bin boundary. Also, it is advisable to give the fitting
%    procedure atleast two bins to work with. 
%    Note that if 'bmin' value is not one of the elements in 
%    boundaries, blgnormfit chooses the bin boundary which is closest 
%    to the specified value and less than that value. Also, the 
%    Default bmin value is the first bin boundary i.e. 
%    boundaries(1).
%  
% 2. [mu, sig] = blgnormfit(h, boundaries, murng, sigrng, 'fineoff');
%    In the default setting, the blgnormfit uses a coarse-grid search     
%    and then a fine grid search for estimating the mu and sigma 
%    paramters. 'fineoff' option allows to to turn off the fine-grid
%    search, since it could be computationally expensive. 
%
% 3. [mu, sig] = blgnormfit(h, boundaries, murng, sigrng, 'fine', [val1; val2; val3]);
%    The 'fine' option lets you set the parameters for fine-grid
%    search. This option expects a matrix which has three values. 
%    The first two values are for mu and sigma respectively. Each
%    value gives the range such that fine grid search is carried over
%    (coarsemu +/- val1) for mu and (coarsesig +/- val2) for sig, both 
%    in steps of val3. 
%    Here, coarsemu and coarsesig are the estimates found using
%    coarse-grid search. 
%
% Version 1.0 (2012)
% Copyright (C) 2012 Yogesh Virkar (University of Colorado, Boulder)
% Distributed under GNU GPL v3.0
% http://www.gnu.org/copyleft/gpl.html
% BLGNORMFIT comes with ABSOLUTELY NO WARRANTY

bminb = [];
fineoff = 0;
finemat = [];
% ---------------------------------------------------------------
% ---------------Parsing command-line arguments------------------
% ---------------------------------------------------------------
i=1;
while i<=length(varargin)
    argok = 1;
    if(ischar(varargin{i}))
        switch varargin{i}
            case 'fineoff', fineoff = 1;
            case 'fine', finemat = varargin {i+1}; i+1;
            case 'bmin', bminb = varargin{i+1}; i=i+1;
            otherwise, argok=0;    
        end
    end
    if ~argok,
        disp(['(BLGNORMFIT) Ignoring invalid argument #' num2str(i+2)]); 
    end
    i=i+1;
end

% ---------------------------------------------------------------
% ------------------------Checking input-------------------------
% ---------------------------------------------------------------

% 1. h must have integer counts.
if isequal(fix(h),h)==0
    fprintf('(BLGNORMFIT) Error: Vector h should be an integer vector.\n');
    return;
end

% 2. h must be non-negative
if ~isempty(find(h<0, 1))
    fprintf('(BLGNORMFIT) Error: Vector h should be non-negative.\n');
    return;
end

% 3. boundaries must have number of elements as one more than 
%    the number in h
if numel(boundaries)~=(numel(h)+1)
    fprintf('(BLGNORMFIT) Error: Incorrect number of elements in either boundaries or h.\n');
    return;
end

% 4. Need atleast 2 bins to work with.
if numel(h)<2
    fprintf('(BLGNORMFIT) Error: I need atleast 2 bins to make this work.\n');
    return;
end

% 5. Checking range vectors
if ~isempty(finemat) && (~isvector(finemat) || min(finemat)<=0 || numel(finemat)~=3)
    fprintf('(BLGNORMFIT) Error: ''fine'' argument must contain a valid vector; using default.\n');
    finemat = [3; 1; 0.01];
end

% 6. Checking bmin option
if ~isempty(bminb) && (~isscalar(bminb) || bminb>=boundaries(end-1))
    fprintf('(BLGNORMFIT) Error: ''bmin'' argument must be a positive value < boundaries(end-1); using default.\n');
    bminb = [];
end

% ---------------------------------------------------------------
% ---------------------------------------------------------------
% ---------------------------------------------------------------

fprintf('Log normal distribution, parameter calculation\n');
fprintf('    Copyright 2012 Yogesh Virkar\n');
fprintf('    Warning: This can be a slow calculation; please be patient.\n');


% Reshape the input vectors
h = reshape(h, numel(h), 1);
boundaries = reshape(boundaries, numel(boundaries), 1);

% Default bmin used is boundaries(1)
if isempty(bminb)
    bminb = boundaries(1);
end
% Default finemat 
if isempty(finemat)
    finemat = [3; 1; 0.05];
end

if isempty(murng)
    murng = -50:1:50;
end
if isempty(sigrng)
    sigrng = 1:0.1:20;
end

ind = find(boundaries<=bminb, 1, 'last');
bminb = boundaries(ind);
h2 = h(ind:end);
boundaries2  = boundaries(ind:end);

l = boundaries2(1:end-1);
u = boundaries2(2:end);

% ---------------------------------------------------------------
% ----------Find parameters using coarse grid search-------------
% ---------------------------------------------------------------

murng = reshape(murng, 1 , numel(murng));
sigrng = reshape(sigrng, 1, numel(sigrng));
MU = repmat(repmat(murng, numel(l), 1), [1, 1, numel(sigrng)]);
temp  = reshape(sigrng, [1 1 numel(sigrng)]);
SIG = repmat(temp, numel(l), numel(murng));
L = repmat(repmat(l, 1, numel(murng)), [1, 1, numel(sigrng)]);
U = repmat(repmat(u, 1, numel(murng)), [1, 1, numel(sigrng)]);
H = repmat(repmat(h2, 1, numel(murng)), [1, 1, numel(sigrng)]);
temp = sum(H.*log((erf((MU-log(L))./(sqrt(2).*SIG))-erf((MU-log(U))./(sqrt(2).*SIG)))./(1+erf((MU-log(bminb))./(sqrt(2).*SIG)))));
fval = reshape(-sum(temp,1), numel(murng), numel(sigrng)); 
clear ('temp', 'L', 'U', 'H', 'MU', 'SIG');

[C, I] = min(fval);
[~, I2] = min(C);
coarsemu = murng(I(I2));
coarsesig = sigrng(I2);

% ---------------------------------------------------------------
% -------------Fine grid search to improve accuracy!------------- 
% ---------------------------------------------------------------
if ~fineoff
    murng = (coarsemu-finemat(1)):finemat(3):(coarsemu+finemat(1));
    sigrng = (coarsesig-finemat(2)):finemat(3):(coarsesig+finemat(2));
    MU = repmat(repmat(murng, numel(l), 1), [1, 1, numel(sigrng)]);
    temp  = reshape(sigrng, [1, 1, numel(sigrng)]);
    SIG = repmat(temp, numel(l), numel(murng));
    L = repmat(repmat(l, 1, numel(murng)), [1, 1, numel(sigrng)]);
    U = repmat(repmat(u, 1, numel(murng)), [1, 1, numel(sigrng)]);
    H = repmat(repmat(h2, 1, numel(murng)), [1, 1, numel(sigrng)]);
    temp = sum(H.*log((erf((MU-log(L))./(sqrt(2).*SIG))-erf((MU-log(U))./(sqrt(2).*SIG)))./(1+erf((MU-log(bminb))./(sqrt(2).*SIG)))));
    fval = reshape(-sum(temp,1), numel(murng), numel(sigrng)); 
    clear ('temp', 'L', 'U', 'H', 'MU', 'SIG');

    [C, I] = min(fval);
    [~, I2] = min(C);
    mu_est = murng(I(I2));
    sigma_est = sigrng(I2);
    
else
    mu_est = coarsemu;
    sigma_est = coarsesig;
end

% Computing likelihood under fitted model (mu_est, sigma_est)
L = -sum( h2.*log((erf((mu_est-log(l))./(sqrt(2)*sigma_est))-erf((mu_est-log(u))./(sqrt(2)*sigma_est)))./(1+erf((mu_est-log(bminb))./(sqrt(2).*sigma_est)))) );

end