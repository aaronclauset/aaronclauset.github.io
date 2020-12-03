function [p, d] = bplpva(h, boundaries, bmin, varargin)

% BPLPVA calculates the p-value for the given power-law fit to some data.
% Source: http://tuvalu.santafe.edu/~aaronc/powerlaws/bins/
%
%   When using binned data, the data vector 'h' is assumed to 
%   contain histogram counts between bin edges 'boundaries'.
%   Usage: a = plvar([900 90 9], [1 10 100 1000])
%   Note that while the above example uses logarithmic binning 
%   (powers of 10), any other binning scheme can be used in 
%   practice. 
%
%----------
% Options:
%----------
% 1. a = bplpva(h, boundaries, bmin, 'range', 1.5:0.01:3.5);
%    The 'range' option can be specified to restrict search for
%    alpha parameter. In above example, bplpva gives the best 
%    looking alpha in the specified range. By default bplpva uses 
%    matlab's fminsearch function which in turn uses the    
%    Nelder-Mead simplex search algorithm. Refer following url:
%    (http://www.mathworks.com/help/techdoc/math/bsotu2d.html#bsgpq6p-11)
%
% 2. a = bplpva(h, boundaries, bmin, 'limit', 100);
%    The 'limit' option lets you limit the search for bmin. 
%    Values in boundaries above this limit are not considered as
%    candidate bmin values. 
%
% 3. a = bplpva(h, boundaries, bmin, 'bmin', 100);
%    The 'bmin' option lets you fix a value such as 100 for bmin. 
%    Note that this value should be one of the values in the 
%    boundaries array. In the above example, 100 cannot be the 
%    last bin boundary. Also, it is advisable to give the fitting
%    procedure atleast two bins to work with. 
%
%    With options 2.and 3., if 'limit' or 'bmin' value is not one 
%    of the elements in boundaries, bplpva chooses the bin boundary
%    which is closest to the specified value and less than that 
%    value.
%  
% 4. a = bplpva(h, boundaries, bmin, 'reps', 10000);
%    The default number of repetitions of fitting procedure is 
%    1000. This number can be changed using the reps option as shown
%    above
%
% 5. a = bplpva(h, boundaries, bmin, 'silent');
%    This option can be used to silence the textual output on 
%    screen.
%
% See also BPLFIT, BPLVAR
%
% Version 1.0 (2012)
% Copyright (C) 2012 Yogesh Virkar (University of Colorado, Boulder)
% Distributed under GNU GPL v3.0
% http://www.gnu.org/copyleft/gpl.html
% BPLPVA comes with ABSOLUTELY NO WARRANTY

rngal = [];
limit = [];
bminb = [];
reps = 1000;
silent = 0;

% ---------------------------------------------------------------
% ---------------Parsing command-line arguments------------------
% ---------------------------------------------------------------
i=1;
while i<=length(varargin)
    argok = 1;
    if(ischar(varargin{i}))
        switch varargin{i}
            case 'range', rngal = varargin{i+1}; i=i+1;
            case 'limit', limit = varargin{i+1}; i=i+1;
            case 'bmin', bminb = varargin{i+1}; i=i+1;
            case 'reps', reps = varargin{i+1}; i=i+1;
            case 'silent', silent = 1;
            otherwise, argok=0;    
        end
    end
    
    if ~argok,
        disp(['(BPLPVA) Ignoring invalid argument #' num2str(i+2)]); 
    end
    i=i+1;
end

% ---------------------------------------------------------------
% ------------------------Checking input-------------------------
% ---------------------------------------------------------------

% 1. h must have integer counts.
if isequal(fix(h),h)==0
    fprintf('(BPLPVA) Error: Vector h should be an integer vector.\n');
    return;
end

% 2. h must be non-negative
if ~isempty(find(h<0, 1))
    fprintf('(BPLPVA) Error: Vector h should be non-negative.\n');
    return;
end

% 3. boundaries must have number of elements as one more than 
%    the number in h
if numel(boundaries)~=(numel(h)+1)
    fprintf('(BPLPVA) Error: Incorrect number of elements in either boundaries or h.\n');
    return;
end

% 4. Need atleast 2 bins to work with.
if numel(h)<2
    fprintf('(BPLPVA) Error: I need atleast 2 bins to make this work.\n');
    return;
end

% 5. Checking range vector
if ~isempty(rngal) && (~isvector(rngal) || min(rngal)<1)
    fprintf('(BPLPVA) Error: ''range'' argument must contain a valid vector; using default.\n');
    rngal = 1.5:0.01:3.5;
end

% 6. Checking limit option
if ~isempty(limit) && (~isscalar(limit) || limit<min(boundaries))
    fprintf('(BPLPVA) Error: ''limit'' argument must be a positive value >= boundaries(1); using default.\n');
    limit = boundaries(end-2);
end

% 7. Checking bmin option
if ~isempty(bminb) && (~isscalar(bminb) || bminb>=boundaries(end-1))
    fprintf('(BPLPVA) Error: ''bmin'' argument must be a positive value < boundaries(end-1); using default.\n');
    bminb = boundaries(1);
end

% 8. Checking number of repititons
if ~isempty(reps) && (~isscalar(reps) || reps<2),
	fprintf('(BPLPVA) Error: ''reps'' argument must be a positive value > 1; using default.\n');
    reps = 1000;
end;

% ---------------------------------------------------------------
% ---------------------------------------------------------------
% ---------------------------------------------------------------

% Reshape the input vectors
h = reshape(h, numel(h), 1);
boundaries = reshape(boundaries, numel(boundaries), 1);

N = sum(h);
d = zeros(reps,1);


if ~silent
    fprintf('Power-law distribution, parameter uncertainty calculation\n');
    fprintf('    Copyright 2012 Yogesh Virkar\n');
    fprintf('    Warning: This can be a slow calculation; please be patient.\n');
    fprintf('    reps=%i\n', length(d));
end

% ---------------------------------------------------------------
%---------------Compute the empirical distance D*------------------
% ---------------------------------------------------------------
% Data above bmin
ind = find(boundaries>=bmin, 1);
z = h(ind:end);     nz = sum(z);
b = boundaries(ind:end);
l = b(1:end-1);
u = b(2:end);

% Data below bmin
y = h(1:ind-1);     %ny = sum(y);
by = boundaries(1:ind);
ly = by(1:end-1);
uy = by(2:end);

% Compute alpha using numerical maximization
hnd = @(alpha) -sum( z.*( log((l).^(1-alpha) - (u).^(1-alpha)) + (alpha-1)*log(bmin) ) );
alpha = fminsearch(hnd, 1);

% Compute distance using KS statistic
temp = cumsum(z(end:-1:1));
cx = 1 - temp(end:-1:1)./nz;
cf = 1 - (l./bmin).^(1-alpha); 
Dstar = max(abs(cf-cx));    

% ---------------------------------------------------------------
% Compute the distribution of gofs using semiparametric bootstrap
% ---------------------------------------------------------------

% Probability of choosing value above bmin 
pz = nz/N;

for i=1:reps
    %semi-parametric bootstrap of data
    n1 = sum(rand(1,N)>pz);
    temp = (ly+uy)./2;
    temp2=[];
    for t=1:numel(y)
        temp2 = [temp2;repmat(temp(t),y(t),1)];
    end
    temp2 = temp2(randperm(numel(temp2)));
    x1 = temp2(ceil(numel(temp2)*rand(n1,1)));
    n2 = N-n1;
    x2 = bmin.*(1-rand(n2,1)).^(-1/(alpha-1));
    x = [x1;x2];
    h2 = histc(x, boundaries);
    h2(end)=[];
    ind = find(h2(end:-1:1)~=0,1,'first')-1;
    if(ind==1)
        h2(end)= [];
    else
        if(ind>1)
            ind2=ind-1;
            h2(end-ind2:end) = [];
        end    
    end
    boundaries2 = boundaries(1:end-ind);
    
    % Need a minimum of 2 bins.
    bmins = boundaries2(1:end-2);
    if ~isempty(bminb)
        bmins = bmins(find(bmins<=bminb, 1, 'last'));
    end
    if ~isempty(limit)
        bmins(bmins>limit) = [];
    end
    dat = zeros(size(bmins));

    for xm=1:length(bmins)
            
        bminq = bmins(xm);
    
        % Truncate the data below bmin
        indq = find(boundaries2>=bminq, 1);
        zq = h2(indq:end);
        nq = sum(zq);
        bq = boundaries2(indq:end); 
            
        % estimate alpha using specified range or using 
        % numerical maximization
        lq = bq(1:end-1);
        uq = bq(2:end);
        if ~isempty(rngal) 
            H = repmat(zq, 1, numel(rngal));        
            LOWER_EDGE = repmat(lq, 1, numel(rngal));
            UPPER_EDGE = repmat(uq, 1, numel(rngal));
            ALPHA_EST = repmat(rngal, numel(bq)-1, 1);
            tempq = H .* (log(LOWER_EDGE.^(1-ALPHA_EST) - UPPER_EDGE.^(1-ALPHA_EST)) + (ALPHA_EST-1) .* log(bminq));
            sum_ = sum(tempq, 1);
            [~,I] = max(sum_);
            al = rngal(I);
        else 
            hnd = @(al2) -sum( zq.*( log((lq).^(1-al2) - (uq).^(1-al2)) + (al2-1)*log(bminq) ) );
            al = fminsearch(hnd, 1);
        end
    
        % compute KS statistic
        tempq = cumsum(zq(end:-1:1));
        cxq = 1 - tempq(end:-1:1)./nq;
        cfq = 1 - (lq./bminq).^(1-al); 
        dat(xm) = max(abs(cfq-cxq));
    end
    
    if ~silent
        fprintf('itr=%i\n', i);
    end
    d(i) = min(dat);
    
end

p = sum(d>=Dstar)./reps;


end