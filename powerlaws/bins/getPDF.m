function prden = getPDF(boundaries, bmin, type, varargin)

% GETPDF calculates the probability density function for specified
% 'type' of model. As of now it calculates PDFs for following
% types of distributions: 
% (1) power law (2) exponential (3) stretched exponential
% (4) log normal (5) power law with exponential cut off. 
% % Source: http://tuvalu.santafe.edu/~aaronc/powerlaws/bins/
%
% ------
% Notes:
% ------
% 1. boundaries: edges of the binned data.
%
% 2. bmin: the lower bound above which you fit any model. 
%
% 3. type: String consisting of one of the five valid types shown 
%          below: 
%          (1) 'pl', (2) 'expn', (3) 'stexp', (4) 'lgnorm', 
%          (5) 'plcut'.
%
% 4. varargin: Depending on the type, GETPDF expects the number of 
%              parameters. 
%              (1) for 'pl', varargin{1} = alpha
%              (2) for 'expn', varargin{1} = lambda
%              (3) for 'stexp', varargin{1} = lambda, varargin{2} = beta
%              (4) for 'lgnorm', varargin{1} = mu, varargin{2} = sigma
%              (5) for 'plcut',  varargin{1} = alpha, varargin{2} = lambda
%
% 5. prden: estimated probability density.
%
% Version 1.0 (2012)
% Copyright (C) 2012 Yogesh Virkar (University of Colorado, Boulder)
% Distributed under GNU GPL v3.0
% http://www.gnu.org/copyleft/gpl.html
% GETPDF comes with ABSOLUTELY NO WARRANTY

% ---------------------------------------------------------------
% ------------------------Checking input-------------------------
% ---------------------------------------------------------------

if numel(boundaries) < 3
    fprintf('(GETPDF) Error: I need atleast 3 boundaries or 2 bins! .\n');
    return;
end


if ~isempty(bmin) && (~isscalar(bmin) || bmin>=boundaries(end-1))
    fprintf('(GETPDF) Error: ''bmin'' argument must be a positive value < boundaries(end-1); using default.\n');
    bmin = boundaries(1);
end
% ---------------------------------------------------------------
% ---------------------------------------------------------------
% ---------------------------------------------------------------

ind = find(boundaries<=bmin, 1, 'last');
bmin2 = boundaries(ind);
l = boundaries(ind:end-1);
u = boundaries(ind+1:end);


switch type
    % Power law
    case 'pl'
        alpha = varargin{1};
        prden = (bmin2^(alpha-1)).*(l.^(1-alpha) - u.^(1-alpha));
    
    % Exponential
    case 'expn'
        lambda = varargin{1};
        prden = exp(lambda.*(bmin2-l)) - exp(lambda.*(bmin2-u));
    
    % Stretched Exponential
    case 'stexp'
        lambda = varargin{1}; beta = varargin{2};
        prden = exp(lambda.*(bmin2^beta - l.^beta)) - exp(lambda.*(bmin2^beta - u.^beta));
    
    % Log normal
    case 'lgnorm'
        mu = varargin{1}; sig = varargin{2};
        lEdge = erf((mu-log(l))/(sqrt(2)*sig));
        uEdge = erf((mu-log(u))/(sqrt(2)*sig));
        cnstC = 1+erf((mu-log(bmin2))/(sqrt(2)*sig));
        prden = (lEdge-uEdge)/(cnstC);
    
    % Power law with exponential cut off
    case 'plcut'
        alpha = varargin{1}; lambda = varargin{2};
        a = 1-alpha;
        x = lambda*bmin2;
        cnst_C = quadgk(@(t)t.^(a-1).*exp(-t),x,inf,'abstol',1e-12,'reltol', 1e-12, 'MaxIntervalCount', 10000); 
        a = 1-alpha;
        x = lambda*u;
        uEdge = zeros(numel(x), 1);
        for ii=1:numel(x)
            uEdge(ii) = quadgk(@(t)t.^(a-1).*exp(-t),x(ii),inf,'abstol',1e-12,'reltol',1e-12, 'MaxIntervalCount', 10000); 
        end
        a = 1-alpha;
        x = lambda*l;
        lEdge = zeros(numel(x), 1);
        for ii=1:numel(x)
            lEdge(ii) = quadgk(@(t)t.^(a-1).*exp(-t),x(ii),inf,'abstol',1e-12,'reltol',1e-12, 'MaxIntervalCount', 10000); 
        end
        prden = (lEdge-uEdge)/cnst_C;
    
    otherwise, fprintf('(GETPDF) Error: Invalid type argument'); return;
end

prden = reshape(prden, numel(prden), 1);

end