function bplplot(h, boundaries, bmin, alpha)

% BPLPLOT visualizes a power-law distributional model with 
% binned empirical data.
% Source: http://www.santafe.edu/~aaronc/powerlaws/bins
%
% Example: Assuming the given binned empirical data is represented
% by bin edges, [1 10 100] and histogram counts, [90 9],
%
% [alpha, bmin, L] = bplfit([90 9], [1 10 100])
% alpha and bmin would be required parameters for power-law model
% found by MLE. 
%
% To visualize the fitted model (alpha, bmin) along with binned 
% empirical data use this function like so,
% 
% plplot([90 9], [1 10 100], bmin, alpha);
%
% See also BPLFIT
%
% Version 1.0 (2012)
% Copyright (C) 2012 Yogesh Virkar (University of Colorado, Boulder)
% Distributed under GNU GPL v3.0
% http://www.gnu.org/copyleft/gpl.html
% BPLPLOT comes with ABSOLUTELY NO WARRANTY

% ---------------------------------------------------------------
% ------------------------Checking input-------------------------
% ---------------------------------------------------------------

% 1. h must have integer counts.
if isequal(fix(h),h)==0
    fprintf('(BPLPLOT) Error: Vector h should be an integer vector.\n');
    return;
end

% 2. h must be non-negative
if ~isempty(find(h<0, 1))
    fprintf('(BPLPLOT) Error: Vector h should be non-negative.\n');
    return;
end

% 3. boundaries must have number of elements as one more than 
%    the number in h
if numel(boundaries)~=(numel(h)+1)
    fprintf('(BPLPLOT) Error: Incorrect number of elements in either boundaries or h.\n');
    return;
end

% ---------------------------------------------------------------
% ---------------------------------------------------------------
% ---------------------------------------------------------------


% Reshape the input vectors
h = reshape(h, numel(h), 1);
boundaries = reshape(boundaries, numel(boundaries), 1);

% Handles for the figures
% hnd = zeros(2,1);

% Empirical CCDF
temp = cumsum(h(end:-1:1));
ch = temp(end:-1:1)./sum(h);

% Fit CCDF
ind = find(boundaries==bmin);
b = boundaries(ind:end);
l2 = b(1:end-1);
cf = (l2./bmin).^(1-alpha);
tmp = cf(1)/ch(ind);
cf = cf./tmp;

% Figure parameters
fntsize = 25;            % Font size
mkrwid = 2;              % Marker width 
mkrsize = 12;            % Marker size

clrdat = [0 0.5 0.7];    % Color for data
clrfit = [0.1 0.1 0.1];  % Color for fit 
lndat = 5;               % Line width for data
lnfit = 3;               % Line width for fit

% Draw figure
figure;
set(gca, 'fontsize', fntsize, 'yscale', 'log', 'xscale', 'log');
xlabel('x', 'fontsize', fntsize);
ylabel('Pr(X \geq x)', 'fontsize', fntsize);

% Plotting empirical data
x1 = boundaries(1); y1 = ch(1);
x2 = boundaries(2); y2 = ch(1);
bi = 2;
ci = 2;
cnt = 2;
while (1)
    hnd1 = line([x1, x2], [y1, y2], 'Color', clrdat);
    set(hnd1, 'linewidth', lndat);     
    hold on;
    hnd2 = line([x1, x2], [y1, y2], 'Color', clrdat);
    set(hnd2, 'linewidth', mkrwid, 'linestyle', 'none', 'marker', 'o', 'markersize', mkrsize, 'markerfacecolor', [1 1 1]);
    hold on;
    x1 = x2; y1 = y2;
    x2 = boundaries(bi); y2 = ch(ci);
    cnt = cnt+1;
    if(mod(cnt,2)==0)
        ci = ci+1;
    else
        bi = bi+1;
    end
    if x2==boundaries(end)
        break;
    end
end
hnd1 = line([x1, x2], [y1, y2], 'Color', clrdat);
set(hnd1, 'linewidth', lndat);
hold on;
hnd1 = line([x1, x2], [y1, y2], 'Color', clrdat);
set(hnd1, 'linewidth', mkrwid, 'linestyle', 'none', 'marker', 'o', 'markersize', mkrsize, 'markerfacecolor', [1 1 1]);

set(gca, 'Linestyleorder', '-'); 
hold on;

% Plotting the power-law fit
x1 = b(1); y1 = cf(1);
x2 = b(2); y2 = cf(1);
bi = 2;
ci = 2;
cnt = 2;
while (1)
    hnd1 = line([x1, x2], [y1, y2], 'Color', clrfit);
    set(hnd1, 'linewidth', lnfit);
    x1 = x2; y1 = y2;
    x2 = b(bi); y2 = cf(ci);
    cnt = cnt+1;
    if(mod(cnt,2)==0)
        ci = ci+1;
    else
        bi = bi+1;
    end
    if x2==b(end)
        break;
    end
end
hnd1 = line([x1, x2], [y1, y2], 'Color', clrfit);
set(hnd1, 'linewidth', lnfit);
    
box on;
hold off;



end