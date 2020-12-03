function [ranks,rho,F,prho,pi,piu]=mvrsample(A, varargin)
% MVRSAMPLE samples the minimum violation rankings for a directed network.
%    Source: http://santafe.edu/~aaronc/facultyhiring/
%    
%    [ranks,rho,F,prho,pi,piu]=mvrsample(A, varargin)
% 
%    MVRSAMPLE(A) samples the minimum violation rankings for a directed 
%    network, based method described in Clauset, Arbesman, Larremore (2015).
%    A is a matrix of size N x N representing a directed network. Each
%    element A(i,j) should be a natural number. Self-loops are allowed.
%    MVRSAMPLE automatically detects whether A meets the requirements of
%    the method.
%   
%    The MVR sampling procedure works as follows:
%    1) Initially, nodes are ranked in descreasing order of their out- 
%       degree. The score of this ranking 'pi' is the number of directed 
%       edges (u,v) for which pi(v) is better than pi(v).
%    2) We then search over rankings to minimize the number of such 
%       violations. The algorithm used is a zero-temperature Markov chain
%       Monte Carlo (MCMC) sampler, which has several user-definable
%       parameters that specify length of burn in, number of samples to
%       take, and spacing between samples.
%    3) By default, this procedure is performed once. Optionally, it can be
%       performed multiple times and the results averaged. Each repetition
%       can be run on a bootstrap of the edges, which would better capture
%       natural variability in the edge generating process (assuming edges 
%       can be treated iid).
%    4) The algorithm returns several outputs that capture the results of
%       the sampler: the average ranking and rho across all repetitions, 
%       and optionally the individual rankings and rhos for each
%       repetition.
%
%
%    Example: 
%       A = rand(100,100)<0.1;
%       names = (1:n)';
%       [ranks,rho,F] = mvrsample(A,'names',names);
%       names(ranks(:,1))
%
%       [ranks,rho,F] = mvrsample(A,'names',names,'bootstrap','reps',10);
%       
%
%    Outputs:
%    'ranks' is a N x 2 matrix whose rows give the (index, mean pi
%            score) for each node in A, in descending order of mean pi, 
%            across repetitions.
%    'rho' is a scalar that gives the fraction of edges in A that violate 
%          the ranks ordering.
%    'F' is a N x N matrix that is equivalent to A under the ranking given
%        by ranks.
%    'prho' is a 1 x N vector containing the rho values for each rep
%    'pi' is a reps x N matrix containing the pi vectors for each rep
%    'piu' is a reps x N matrix containing the stddev of pi for each rep
%    
%    
%    For more information, try 'type mvrsample'

% Version 1.0    (2015 February)
% Copyright (C) 2015 Aaron Clauset (University of Colorado Boulder)
% Distributed under GPL 2.0
% http://www.gnu.org/licenses/gpl-2.0.html
% MVRSAMPLE comes with ABSOLUTELY NO WARRANTY
% 
% Notes:
% 
% 1. The default 'move' in the MCMC is to choose a pair of nodes and
%    propose a ranking in which they are 'rotated'. The number of nodes in
%    the set to be rotated is an adjustable parameter. For instance, this
%    uses 3-node rotations:
%    
%       [ranks,rho,F] = mvrsample(A,'rotate',3);
%    
% 2. Each repetition can be performed on a simple bootstrap of the edges.
%    The adjacency matrix is interpreted as an unweighted, directed
%    multigraph, in which A(i,j) = g indicates that there are g unweighted
%    edges (i,j) in the network. The bootstrap chooses m=sum(sum(A)) edges
%    with replacement from the set of all such edges (i,j) and builds a new
%    network from these. The default setting is no bootstrap. To turn it
%    on, simply call
%    
%       [ranks,rho,F] = mvrsample(A,'bootstrap');
% 
% 3. The parameters of the MCMC can also be changed: the number of samples 
%    to take before stopping, the number of swaps to successfully perform 
%    between each sampled ranking, and the number of steps used for 'burn
%    in' before the sampling begins. These can be changed like so
%    
%       [ranks,rho,F] = mvrsample(A,'samples',1000,'spacing',10000,'burnin',10^6);
%
%    Any subset of these can be omitted, which will leave them at their
%    default settings.
%
% 4. The algorithm writes updates about the progress of the MCMC to stdout.
%    This can be silenced like so
%    
%       [ranks,rho,F] = mvrsample(A,'quiet');
%
%

names  = [];        % names of the nodes
nr     = [];        % size of a rotation
bstrap = [];        % boostrap the edges?
nreps  = [];        % number of repititions of sampler to run
s_rate = [];        % take sample every s_rate steps 
n_samp = [];        % number of samples to store
Tb     = [];        % burn-in steps before sampling
nowarn = false;     % default: don't display warnings
quiet  = false;     % default: display messages

% re-initialize random number generator at most once this session
persistent rand_state;
if isempty(rand_state)
    rand_state = cputime;
    rand('twister',sum(100*clock));
end;

% parse command-line parameters; trap for bad input
i=1; 
while i<=length(varargin), 
  argok = 1; 
  if ischar(varargin{i}), 
    switch varargin{i},
        case 'names',        names   = varargin{i+1}; i = i + 1;
        case 'rotate',       nr      = varargin{i+1}; i = i + 1;
        case 'reps',         nreps   = varargin{i+1}; i = i + 1;
        case 'samples',      n_samp  = varargin{i+1}; i = i + 1;
        case 'spacing',      s_rate  = varargin{i+1}; i = i + 1;
        case 'burnin',       Tb      = varargin{i+1}; i = i + 1;
        case 'bootstrap',    bstrap  = true;
        case 'nowarn',       nowarn  = true;
        case 'quiet',        quiet   = true;
        otherwise, argok=0; 
    end
  end
  if ~argok, 
    disp(['(MVRSAMPLE) Ignoring invalid argument #' num2str(i+1)]); 
  end
  i = i+1; 
end
if isempty(A) || (~isempty(A) && (size(A,1) ~= size(A,2))),
	fprintf('(MVRSAMPLE) Error: input ''A'' must be a non-empty square matrix; bailing out.\n');
    return;
end;
if any(any(A<0)) || ~isempty(setdiff(A,floor(A))),
	fprintf('(MVRSAMPLE) Error: input ''A'' elements must be natural numbers; bailing out.\n');
    return;
end;
if ~isempty(names) && (numel(names) ~= size(A,1)),
	fprintf('(MVRSAMPLE) Error: ''names'' argument must be same length as size(A,1); using default.\n');
    names = [];
end;
if ~isempty(nr) && ((~isscalar(nr) || nr<2) && isempty(setdiff(nr,floor(nr)))),
	fprintf('(MVRSAMPLE) Error: ''rotate'' argument must be a positive integer > 1; using default = 2.\n');
    nr = [];
end;
if ~isempty(nreps) && ((~isscalar(nreps) || nreps<1) && isempty(setdiff(nreps,floor(nreps)))),
	fprintf('(MVRSAMPLE) Error: ''reps'' argument must be a positive integer > 0; using default = 100.\n');
    nreps = [];
end;
if ~isempty(n_samp) && ((~isscalar(n_samp) || n_samp<1) && isempty(setdiff(n_samp,floor(n_samp)))),
	fprintf('(MVRSAMPLE) Error: ''samples'' argument must be a positive integer > 0; using default = 2000.\n');
    n_samp = [];
end;
if ~isempty(s_rate) && ((~isscalar(s_rate) || s_rate<1) && isempty(setdiff(s_rate,floor(s_rate)))),
	fprintf('(MVRSAMPLE) Error: ''spacing'' argument must be a positive integer > 0; using default = 10000.\n');
    s_rate = [];
end;
if ~isempty(Tb) && ((~isscalar(Tb) || Tb<1) && isempty(setdiff(Tb,floor(Tb)))),
	fprintf('(MVRSAMPLE) Error: ''burnin'' argument must be a positive integer > 0; using default = 10^6.\n');
    Tb = [];
end;

% basic network statistics
n  = size(A,1);
m  = sum(sum(A));

% default settings
if isempty(nr),     nr = 2;            end; % rotation size = pairwise swaps
if isempty(names),  names = (1:n)';    end; % names = indices
if isempty(bstrap), bstrap = false;    end; % bootstrap = false
if isempty(nreps),  nreps = 1;         end; % 1 repetition of sampler
if isempty(s_rate), s_rate = n;        end; % n steps per sampled state
if isempty(n_samp), n_samp = n;        end; % n samples stored
if isempty(Tb),     Tb = n^2;          end; % n^2 steps for burn-in

tstr = {'off' 'on'};
if ~quiet,
    fprintf('Minimum violation ranking sampler\n');
    fprintf('   Copyright 2015 Aaron Clauset\n');
    fprintf('   Warning: This can be a very slow calculation; please be patient.\n');
    fprintf('   nodes, n = %i\n   edges, m = %i\n   reps     = %i\n',n,m,nreps);
    fprintf('   bootstrap of edges      = %s\n',tstr{bstrap+1});
    fprintf('   number of nodes to swap = %i\n',nr);
    fprintf('   steps between samples   = %i\n',s_rate);
    fprintf('   target sample count     = %i\n',n_samp);
end;

tic;                   % start the clock
prho = zeros(1,nreps); % output: fraction of edges that violate MVR (by rep)
pi   = zeros(n,nreps); % output: mean of ranks across MVR samples (by rep)
piu  = zeros(n,nreps); % output: std of ranks across MVR samples (by rep)

for ijk=1:nreps

    % 1. if necessary, bootstrap the set of edges by sampling them
    %    uniformly at random with replacement. turning this feature off
    %    will reduce the sampler's ability to accurately estimate the
    %    uncertainty of the MVR score.
    if bstrap==true
        % 1a. bootstrap the edges
        [u, v, s] = find(sparse(A));
        m       = sum(s);
        adj     = zeros(m,3);
        k       = 1;
        for i=1:length(u)
            adj(k:k+s(i)-1,:) = repmat([u(i) v(i) 1],s(i),1);
            k = k + s(i);
        end;
        bts = adj(ceil(m.*rand(m,1)),:);
        B   = zeros(size(A)); % adjacency matrix, bootstrapped
        for i=1:size(bts,1)
            ii=bts(i,1);
            jj=bts(i,2);
            B(ii,jj) = B(ii,jj)+1;
        end;
    else
        % 1b. don't bootstrap the edges
        B=A;
    end;


    % 2a. initialize the ranking out-degree, in decreasing order
    h     = zeros(n,1);                  % the ranking vector
    kout  = sum(B,2);                    % get the out-degrees
    [~,I] = sort(kout,'descend');        % sort them
    for i=1:n, h(I(i)) = i; end;         % update the ranking vector

    % 2b. initialize the MVR score
    F = zeros(n,n);    % the reordered adjacency matrix
    for i=1:n
        for j=i:n
            F(h(i),h(j)) = F(h(i),h(j)) + B(i,j);
            if i~=j
                F(h(j),h(i)) = F(h(j),h(i)) + B(j,i);
            end;
        end;
    end;
    score = sum(sum(triu(F,0))) - sum(sum(tril(F,-1)));
    if ~quiet,
        fprintf('[rep=%i][t=%4.0f] violations = %i (%3.1f%%)\tconverging: %i\t(%4.2fm done)\n',ijk,1,m-score,100*(1-score/m),Tb,toc/60);
    end;
    maxs  = score;     % the best score so far

    % 2c. initialize the zero-temperature MCMC sampler
    rs     = zeros(n,n_samp);  % stored samples
    k      = 1;                % index of sample
    T      = Tb+n_samp*s_rate; % 
    f_stop = 0;
    cnt    = 0;
    t      = 1;
    
    % 3. Run the zero-temperature MCMC to sample the minimum violation
    %    rankings. The proposal step of the MCMC chooses a uniformly random
    %    group of vertices of size r and then tries rotating them to create
    %    a new ordering. If that ordering is no worse than the current 
    %    ordering, it accepts the move (Metropolis-Hastings rule) and
    %    repeats. Otherwise, it discards that proposal, and chooses a new
    %    one. The MCMC runs for Tb "burn in" steps before beginning the 
    %    sampling of MVRs. Some information is written to stdout as the
    %    MCMC progresses.
    while true
        t=t+1;
        % 3a. check stopping criteria
        if t>T, f_stop = 1; end;
        if f_stop>0, break; end;
        % 3b. choose r positions to swap
        h2 = h;
        r  = 1+ceil((nr-1)*rand(1,1));
        pr = ones(1,r);
        while length(unique(h2(pr)))<length(pr)
            pr = randi(n,1,r);
        end;
        % 3c. "rotate" them
        h2(pr) = [h2(pr(end)); h2(pr(1:end-1))];
        % 3d. tabulate proposed block matrix
        F2 = zeros(n,n);
        for i=1:n
            for j=i:n
                F2(h2(i),h2(j)) = F2(h2(i),h2(j)) + B(i,j);
                if i~=j
                    F2(h2(j),h2(i)) = F2(h2(j),h2(i)) + B(j,i);
                end;
            end;
        end;
        % 3e. compute F2's score
        snew = sum(sum(triu(F2,0))) - sum(sum(tril(F2,-1)));
        if snew>=maxs
            % if new maximum
            if snew>maxs
                maxs = snew;  score = snew;
                if ~quiet && t>=Tb
                    fprintf('[rep=%i][t=%i] violations = %i (%3.1f%%)\tfound a better MVR; restarting sampling\n',ijk,t,m-score,100*(1-score/m));
                end;
                if t>Tb, [k, cnt, t] = deal(1,0,Tb+1); end; % reset sampling
            end;
            cnt  = cnt + 1; % increment neutral counter
            h    = h2;      % store new ordering
            F    = F2;      % store new ordered adjancecy matrix
        end;
        if t>Tb && mod(t,ceil(s_rate))==0
            rs(:,k) = h;   % store sample
            k       = k+1; % count number of samples
            cnt     = 0;   % reset neutral counter
        end;

        % 3f. update the user on the progress (stdout)
         if mod(t,1000)==0
            if ~quiet,
                if t<= Tb
                    fprintf('[rep=%i][t=%i] violations = %i (%3.1f%%)\tconverging: %i\t(%4.2fm done | %4.2fm to go)\n',ijk,t,m-score,100*(1-score/m),Tb-t,toc/60,((T*nreps)/(t+(ijk-1)*t-1))*(toc/60));
                else
                    fprintf('[rep=%i][t=%i] violations = %i (%3.1f%%)\tsamples: %i (%4.1f%%)\t(%4.2fm done | %4.2fm to go)\n',ijk,t,m-score,100*(1-score/m),k,100*k/n_samp,toc/60,((T*nreps)/(t+(ijk-1)*t-1))*(toc/60));
                end;                
            end;
            % write mean ranks for the top-50 (so far)
            if t>Tb && k>2
                ranks = [(1:n)' mean(rs(:,1:k-1),2) std(rs(:,1:k-1)')'];
                ranks = sortrows(ranks,2);
                for kik=1:50
                    grab = ranks(kik,1);
                    if ~quiet,
                        fprintf('%4.2f (%4.2f)  %s\n',ranks(kik,2),ranks(kik,3),char(names(grab)));
                    end;
                end;
            end;
         end;
        % 3g. recheck stopping criteria
        if t>T, f_stop = 1; end;
        if f_stop>0, break; end;
    end;
    
    % store the results of this rep
    prho(1,ijk) = (m-score)/m;
    pi(:,ijk)  = mean(rs,2);
    piu(:,ijk) = std(rs')';
    
end;

% compute the mean results and return them
ranks = sortrows([(1:n)' mean(pi,2)],2);

[~,I] = sort(mean(pi,2));
for i=1:n, h(I(i)) = i; end;
F = zeros(n,n);    % the reordered adjacency matrix
for i=1:n
    for j=i:n
        F(h(i),h(j)) = F(h(i),h(j)) + A(i,j);
        if i~=j
            F(h(j),h(i)) = F(h(j),h(i)) + A(j,i);
        end;
    end;
end;

% fraction of edges that violate the ranking
rho = sum(sum(tril(F,-1)))/m;

% done.


