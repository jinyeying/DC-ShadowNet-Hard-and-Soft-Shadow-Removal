function [ entropy ] = getEntropy( proj, bias ) 
%we take only 90% of the median data, noise, projected data 
[muhat, ~, ~, sigmaci] = normfit(proj, 0.1);
%MUHAT is an estimate of the mean, and SIGMAHAT is an estimate of the standard deviation.
%returns 100(1-0.1) percent confidence intervals for the parameter estimates.
[~, N] = size(proj);
proj2 = [];
for i = 1:N
    if proj(i) >= muhat-sigmaci(1) && proj(i) <= muhat+sigmaci(2)
        proj2 = [proj2, proj(i)];
    end;
end;
proj = proj2;

% Scott's Rule
[~, N] = size(proj);
binSize = (3.49 * std(proj)) / (nthroot(N,3));%3.49 std(projected_data)N^(1/3)
binNum = ceil(abs((max(proj) - min(proj)) / binSize));

% histogram counting
[c, x] = hist(proj, binNum);

% normalizacia histogram
normalized = c/trapz(x,c);
normalized = normalized / sum(normalized);
logNormalized = arrayfun(@(x) log(x), normalized);

%I check out the values that meet the bias because of the large numbers at the log
[~, num] = size(normalized);
idx = 1;
normalized2 = zeros(1,sum(normalized > bias));
logNormalized2 = zeros(1,sum(normalized > bias));
for i = 1:num
    if normalized(i) > bias
       normalized2(idx) = normalized(i);
       logNormalized2(idx) = logNormalized(i);
       idx = idx + 1;
    end
end

%final entropy
entropyH = normalized2 .* logNormalized2;
entropy = -sum(entropyH);

end

