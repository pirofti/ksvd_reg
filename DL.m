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

function [D, X, err] = DL(Y, D, s, iternum, varargin)
%% Dictionary Learning (DL) Iterations
% INPUTS:
%	Y -- signal set
%   D -- dictionary
%   s -- sparity target
%   iternum -- DL iterations
%
% OUTPUTS:
%   D -- the resulting dictionary
%   X -- the corresponding sparse representations
%   err -- the RMSE at each iteration
    [p, m] = size(Y);
    err = zeros(1,iternum);

    for iter = 1:iternum
        X = omp(D'*Y, D'*D, s, 'checkdict', 'off');
        [D, X] = ksvd_reg(Y, D, X, iter, varargin{:});
        err(iter) = norm(Y - D*X, 'fro') / sqrt(p*m);
    end
end