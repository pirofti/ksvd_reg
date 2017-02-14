% Copyright (c) 2016 Paul Irofti <paul@irofti.net>
% 
% Permission to use, copy, modify, and/or distribute this software for any
% purpose with or without fee is hereby granted, provided that the above
% copyright notice and this permission notice appear in all copies.
% 
% THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
% WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
% MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
% ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
% WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
% ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
% OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

function [D,X] = ksvd_reg(Y,D,X,iter,varargin)
%% Regularized K-SVD algorithm
% INPUTS:
%   Y -- training signals set
%   D -- current dictionary
%   X -- sparse representations
%   iter -- current DL iteration
%
% PARAMETERS:
%   reg -- regularization factor (default: 0.01)
%   vanish -- regularization vanishing factor (default: 0.95)
%   regstop -- stop regularization from this iteration on (default: Inf)
%
% OUTPUTS:
%   D -- updated dictionary
%   X -- updated representations
    persistent mu;
    persistent vanish;
    persistent regstop;
    
    if iter == 1
        p = inputParser();
        p.KeepUnmatched=true;
        p.addParameter('reg', 0.01);
        p.addParameter('vanish', 0.95);
        p.addParameter('regstop', Inf);
        p.parse(varargin{:});
        mu = p.Results.reg;
        vanish = p.Results.vanish;
        regstop = p.Results.regstop;
    else
        if regstop > iter
            mu = mu * vanish;
        else
            mu = 0;
        end
    end

    % Use sparse pattern from OMP to compute new sparse representations
    X = ompreg(Y,D,X,mu);
    
    % Dictionary atom update loop
    for j = 1:size(D,2)     
        [~, data_indices, ~] = find(X(j,:));
        d = D(:,j);
        D(:,j) = 0;
        if (isempty(data_indices))
            D(:,j) = d;
            continue;
        end
        smallX = X(:,data_indices);
        smallY = Y(:,data_indices);
        [D(:,j),X(j,data_indices)] = reg_up(smallY,D,smallX,mu);
    end
end

function [d,x] = reg_up(Y,D,X,mu)
%% K-SVD with regularization atom update
% INPUTS:
%   Y -- training signals set
%   D -- current dictionary
%   X -- sparse representations
%   mu -- regularization factor
%
% OUTPUTS:
%   d -- updated atom
%   x -- updated representations corresponding to the current atom
    [d,s,x] = svds(Y - D*X, 1);
    x = s*x/(1+mu);
end