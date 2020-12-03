function h=pleplot(x, s, varargin)
% PLEPLOT visualizes an ensemble of power-law models with empirical data.
%    Source: http://www.santafe.edu/~aaronc/rareevents/
%
%    Given a input vector of event sizes x and a set of power-law model 
%    parameters s, PLEPLOT(x,s) plots the complementary CDF of the data 
%    and the models on log-log axes. PLEPLOT is compatible with the output
%    of both PLOUT and PLOUTM; in the latter case, each covariate type
%    produces a separate figure.
%    
%    By default, PLEPLOT visualizes 100 models from s, chosen uniformly at 
%    random.
%    
%    PLEPLOT automatically detects the type of input data (discrete or
%    continuous) and plots the appropriate version of the power-law 
%    model. For discrete data, if min(x) > 1000, PLEPLOT uses the continuous
%    approximation, which is reliable in this regime.
%   
%    Example:
%       x = (1-rand(100,1)).^(-1/(2.5-1));
%       [~ ~ s] = plout(x);
%       h = plplot(x,s);                 % default behavior
%       h = plplot(x,s,'models',100);    % plot 100 models
%
%    Outputs:
%     h: contains handles to each of the plotted data series. h(end) is the
%        empirical data.
%    
%    For more information, try 'type pleplot'
%
%    See also PLOUT

% Version 1.0    (2011 December)
% Copyright (C) 2011-2012 Aaron Clauset (Univerity of Colorado, Boulder)
% Distributed under GPL 2.0
% http://www.gnu.org/copyleft/gpl.html
% PLEPLOT comes with ABSOLUTELY NO WARRANTY
% 
% Notes:
% 
% 1. PLEPLOT can be told to plot a specific number of models out of the
%    matrix s:
%    
%       p = pleplot(x,s,'models',100);
%    
% 2. PLEPLOT can be told to omit the legend like so
%    
%       p = pleplot(x,s,'nolegend');
%    
% 3. To force PLEPLOT to use the first k models within s, say
%    
%       p = pleplot(x,s,'firstk');
%    
%    This command can be used in conjunction with 'models' to specify k.
%
% 4. To label the largest event with a vertical line
%    
%       p = pleplot(x,s,'labcat');
%

f_noleg  = false;       % legend is plotted
f_firstk = false;       % choose models uniformly at random
f_labcat = false;       % label largest event
nummod   = 100;         % number of models to plot
persistent rand_state;

% parse command-line parameters; trap for bad input
i=1; 
while i<=length(varargin), 
  argok = 1; 
  if ischar(varargin{i}), 
    switch varargin{i},
        case 'models',       nummod   = varargin{i+1}; i = i + 1; 
        case 'nolegend',     f_noleg  = true;
        case 'firstk',       f_firstk = true;
        case 'labcat',       f_labcat = true;
        otherwise, argok=0; 
    end
  end
  if ~argok, 
    disp(['(PLEPLOT) Ignoring invalid argument #' num2str(i+1)]); 
  end
  i = i+1; 
end

% 1a. -- check structure of inputs x and s
sz = size(x);
if (sz(1)==2 || sz(1)==1), x = x'; elseif (sz(2)==2 || sz(2)==1), x = x; else 
	fprintf('(PLEPLOT) Error: input argument x must by a Nx1 or Nx2 vector; halting.\n');
    return;
end;
if size(s,2)==4 && size(x,2)>1
	fprintf('(PLEPLOT) Error: input argument s missing covariate dimension; halting.\n');
    return;
elseif size(s,2)==5 && size(x,2)==1
	fprintf('(PLEPLOT) Error: input argument x missing covariate dimension; halting.\n');
    return;
end;
if size(s,2)==4 && size(x,2)==1
    s(:,5) = ones(size(s,1),1);
    x(:,2) = ones(size(x,1),1);
end;
    
% 1b. -- check input arguments
if ~isscalar(nummod) || nummod < 1
	fprintf('(PLEPLOT) Error: ''models'' argument must be a natural number; using default.\n');
    nummod = 100;
end;
if nummod > size(s,1)
    nummod   = size(s,1);
    f_firstk = true;
end;

% 1c. -- initialize random number generator
if isempty(rand_state)
    rand_state = cputime;
    rand('twister',sum(100*clock)); % deprecated
end;

% make the plots
h    = zeros(nummod+1,1);       % output handles
c    = unique(x(:,2));          % unique covariate values
u    = length(c);               % number of covariates
% for each covariate type
for ell=1:u
    z  = x(x(:,2)==c(ell),1);
    sz = s(s(:,5)==c(ell),(1:4));
    zmax = 10.^ceil(log10(max(x(:,1))));  %  right-most edge of x-axis
    ymin = 10.^floor(log10(1/size(x,1))); % bottom-most edge of y-axis
    % select discrete or continuous power-law model for this covariate
    if     isempty(setdiff(z,floor(z))), f_dattype = 'INTS';
    elseif isreal(z),    f_dattype = 'REAL';
    end;
    if strcmp(f_dattype,'INTS') && (min(z) > 1000 && length(z)>100),
        f_dattype = 'REAL';
    end;
    switch f_dattype,
        case 'REAL',
            n = size(z,1);
            c = [sort(z) (n:-1:1)'./n];
            figure;
            loglog(c(1,1),c(1,2),'ko','MarkerSize',8); hold on;
            for k=1:nummod
                % choose a model
                if f_firstk, g = k; else g = randi(size(sz,1),1); end;
                [alpha,xmin] = deal(sz(g,1),sz(g,2));
                % make the model
                q = [sort(z(z>=xmin)); zmax];
                cf = [q (q./xmin).^(1-alpha)];
                cf(:,2) = cf(:,2) .* c(find(c(:,1)>=xmin,1,'first'),2);
                % plot the model
                h(k) = loglog(cf(:,1),cf(:,2),'r-','LineWidth',1);
            end;
            % plot the data
            h(end) = loglog(c(:,1),c(:,2),'ko','MarkerSize',8);
            if f_labcat
                loglog(max(x(:,1)).*[1 1],[ymin 1],'k--');
            end;
            hold off;
            % set axes limits and tick marks
            xr  = [10.^floor(log10(min(z))) zmax];
            xrt = (round(log10(xr(1))):2:round(log10(xr(2))));
            if length(xrt)<4, xrt = (round(log10(xr(1))):1:round(log10(xr(2)))); end;
            yr  = [ymin 1];
            yrt = (round(log10(yr(1))):2:round(log10(yr(2))));
            if length(yrt)<4, yrt = (round(log10(yr(1))):1:round(log10(yr(2)))); end;
            set(gca,'XLim',xr,'XTick',10.^xrt);
            set(gca,'YLim',yr,'YTick',10.^yrt,'FontSize',16);
            % label your axes
            ylabel('Pr(X\geq x)','FontSize',16);
            xlabel('x','FontSize',16)
            % legend and title
            if ~f_noleg
                h1 = legend('Empirical data','Power-law models',1);
                set(h1,'FontSize',16);
                if u>1
                    title(['Covariate ' num2str(c(ell))],'FontSize',16);
                end;
            end;

        case 'INTS',
            n = length(z);        
            q = unique(z);
            c = hist(z,q)'./n;
            c = [[q; q(end)+1] 1-[0; cumsum(c)]]; c(c(:,2)<10^-10,:) = [];
            figure;
            loglog(c(1,1),c(1,2),'ko','MarkerSize',8); hold on;
            for k=1:nummod
                % choose a model
                if f_firstk, g = k; else g = randi(size(sz,1),1); end;
                [alpha,xmin] = deal(sz(g,1),sz(g,2));
                % make the model
                cf = ((xmin:zmax)'.^-alpha)./(zeta(alpha) - sum((1:xmin-1).^-alpha));
                cf = [(xmin:zmax+1)' 1-[0; cumsum(cf)]];
                cf(:,2) = cf(:,2) .* c(c(:,1)==xmin,2);
                % plot the model
                h(k) = loglog(cf(:,1),cf(:,2),'r-','LineWidth',1);
            end;
            h(end) = loglog(c(:,1),c(:,2),'ko','MarkerSize',8);
            if f_labcat
                loglog(max(x(:,1)).*[1 1],[ymin 1],'k--');
            end;
            hold off;
            % set axes limits and tick marks
            xr  = [10.^floor(log10(min(z))) zmax];
            xrt = (round(log10(xr(1))):2:round(log10(xr(2))));
            if length(xrt)<4, xrt = (round(log10(xr(1))):1:round(log10(xr(2)))); end;
            yr  = [ymin 1];
            yrt = (round(log10(yr(1))):2:round(log10(yr(2))));
            if length(yrt)<4, yrt = (round(log10(yr(1))):1:round(log10(yr(2)))); end;
            set(gca,'XLim',xr,'XTick',10.^xrt);
            set(gca,'YLim',yr,'YTick',10.^yrt,'FontSize',16);
            % label your axes
            ylabel('Pr(X\geq x)','FontSize',16);
            xlabel('x','FontSize',16)
            % legend and title
            if ~f_noleg
                h1 = legend('Empirical data','Power-law models',1);
                set(h1,'FontSize',16);
                if u>1
                    title(['Covariate ' num2str(c(ell))],'FontSize',16);
                end;
            end;

        otherwise,
            fprintf('(PLEPLOT) Error: x must contain only reals or only integers.\n');
            h = [];
            return;
    end;
end;

