function [normR, p] = blrtest(p1, p2, h, boundaries, bmin, isNested)

% BLRTEST computes the likelihood ratio.
% Source: http://tuvalu.santafe.edu/~aaronc/powerlaws/bins/
%
% -------
% Notes:
% -------
%
% 1. p1, p2: These are the pdfs of the two distributional models 
%            which you want to compare.
%            
% 2. h, boundaries: The given binned empirical data.
%
% 3. isNested: If value of this flag is 1, it indicates that the 
%              two models are nested. Example: isNested=1, if we 
%              have power law and power law with exponential cut
%              off.
%
% 4. normR: gives the normalized likelihood ratio for non-nested
%           models and likelihood ratio for nested models
%
% 5. p: The p-value calculated as per Vuoung's test. If this value
%       is less than 0.1, then sign of normR can be trusted, else
%       the likelihood ratio test is inconclusive. 
%       
% -------
% Usuage: 
% -------
% E.g.: Suppose we want to compare power-law fit (p1) to 
% log-normal fit (p2)
%
% Calculate probability densities p1 and p2 using GETPDF function 
% as shown below,
%
% p1 = getPDF(boundaries, bmin, 'pl', alpha);
% p2 = getPDF(boundaries, bmin, 'lgnorm', mu, sigma);
% 
% Calculate normalized likelihood ratio using,
%
% [normR, p] = blrtest(p1, p2, h, boundaries, bmin, 0);
%
%
% See also GETPDF
%
% Version 1.0 (2012)
% Copyright (C) 2012 Yogesh Virkar (University of Colorado, Boulder)
% Distributed under GNU GPL v3.0
% http://www.gnu.org/copyleft/gpl.html
% BLRTEST comes with ABSOLUTELY NO WARRANTY


% ---------------------------------------------------------------
% ------------------------Checking input-------------------------
% ---------------------------------------------------------------

% 1. h must have integer counts.
if isequal(fix(h),h)==0
    fprintf('(BLRTEST) Error: Vector h should be an integer vector.\n');
    return;
end

% 2. h must be non-negative
if ~isempty(find(h<0, 1))
    fprintf('(BLRTEST) Error: Vector h should be non-negative.\n');
    return;
end

% 3. boundaries must have number of elements as one more than 
%    the number in h
if numel(boundaries)~=(numel(h)+1)
    fprintf('(BLRTEST) Error: Incorrect number of elements in either boundaries or h.\n');
    return;
end

% 4. Need atleast 2 bins to work with.
if numel(h)<2
    fprintf('(BLRTEST) Error: I need atleast 2 bins to make this work.\n');
    return;
end

% ---------------------------------------------------------------
% ---------------------------------------------------------------
% ---------------------------------------------------------------

% Reshape the input vectors
h = reshape(h, numel(h), 1);
boundaries = reshape(boundaries, numel(boundaries), 1);
p1 = reshape(p1, numel(p1), 1);
p2 = reshape(p2, numel(p2), 1);

ind = find(boundaries<=bmin, 1, 'last');
h2 = h(ind:end);
boundaries2 = boundaries(ind:end);

% Make binned data look continuous
l = boundaries2(1:end-1);
u = boundaries2(2:end);
n = sum(h2);
temp = (l+u)./2;
temp2=[];
for y=1:numel(h2)
    temp2 = [temp2;repmat(temp(y),h2(y),1)];
end
N = numel(temp2);    
temp2 = temp2(randperm(N));
[~,whichbin] = histc(temp2, boundaries2);

% Calculate log-likelihood at each data point.
l1 = log(p1(whichbin));
l2 = log(p2(whichbin));

NEGINF = -100000000;
l1(l1==-Inf) = NEGINF;
l2(l2==-Inf) = NEGINF;

% Calculate mean of log likelihoods
l1_bar = (1/n)*sum(l1);
l2_bar = (1/n)*sum(l2);

% Calculate likelihood ratio (LR)
R = sum((l1-l2));


% Computing normalized LR is non-nested. 
if ~isNested
    temp = ((l1-l2)-(l1_bar- l2_bar)).^2;
    sigmaR = sqrt((1/n)*sum(temp(~isnan(temp))));
    % Vuoung test
    p = erfc(abs(R)/(sqrt(2*n)*sigmaR));
    normR = R/(sqrt(n)*sigmaR);
else
    %{
    % mean of the R
    mu_R = l1_bar-l2_bar; 
    ncm2_R  = 3;
    % second central moment for R
    cm2_R = ncm2_R - (mu_R)^2; 
    sigmaR = sqrt(cm2_R);
    normR = R/(sqrt(n)*sigmaR);
    %}
    p = erfc(sqrt(abs(R))/sqrt(2));
    normR = R;
    
end

end