close all;
clear;
clc;
image1 = im2double(imread('.\input\88.png'));%88,27,28,70-5.
%image1 = im2double(imread('.\input\1.jpg'));
I_gray = rgb2gray(image1);
I_gray_valid = I_gray;
I_gray_valid(I_gray>0.9) = 0;   %%remove saturared pixel
I_org = im2double(image1);
myfilter = fspecial('gaussian',[3 3], 0.5);
[h, w, ~] = size(I_org);
I = imfilter(I_org, myfilter, 'replicate');
%remove zeros due to log and division
I(I_org==0) = 1;
IR = I(:, :, 1);
IG = I(:, :, 2);
IB = I(:, :, 3);
SUMRGB = IR + IG + IB;
mean_SUMRGB = (IR + IG + IB)./3;
ch = double(zeros(h, w, 3));
ch(:,:,1) = IR ./ SUMRGB;
ch(:,:,2) = IG ./ SUMRGB;
ch(:,:,3) = IB ./ SUMRGB;

TT = nthroot(IR .* IG .* IB, 3);%3 Cubic root
R = IR ./ TT;
G = IG ./ TT;
B = IB ./ TT;
s = size(R,1) * size(R,2);
RR = reshape(R, 1, s);
GG = reshape(G, 1, s);
BB = reshape(B, 1, s);
RR = arrayfun(@(x) log(x), RR);%log(R ./ TT);
GG = arrayfun(@(x) log(x), GG);%log(G ./ TT);
BB = arrayfun(@(x) log(x), BB);%log(B ./ TT);
v1 = [1/sqrt(2); -1/sqrt(2); 0]';
v2 = [1/sqrt(6); 1/sqrt(6); -2/sqrt(6)]';
U = [v1; v2];
O = [RR; GG; BB];              %%log((R,G,B)/ TT);
res = double(zeros(2,s));
parfor i = 1:s
    res(:,i) = U * O(:,i);
end;
X = res(1,:);
Y = res(2,:);
chromaticityVec = [X; Y];
%%remove gray pixel
I_gray_valid(abs(mean_SUMRGB-IR)<0.008)=0;
I_gray_valid(abs(mean_SUMRGB-IG)<0.008)=0;
I_gray_valid(abs(mean_SUMRGB-IB)<0.008)=0;

entropyBias=0.00001;
bestTheta = 1;
bestEntropy = inf;
bestProj = [];
idx = 1;
[~, num] = size(chromaticityVec);
l_start = 1; l_end = 180; l_step = 1;
entropy = zeros(1, floor((l_end-l_start) / l_step) + 1);
for theta = l_start:l_step:l_end
    x = cos(theta * pi / 180);
    y = sin(theta * pi / 180);
    u = [x; y];                               %projection vector
    proj = zeros(1,num);
    parfor i = 1:num
       proj(i) = dot(chromaticityVec(:,i), u);%cos;sin
    end
    entropy(idx) = getEntropy(proj, entropyBias);

    if(entropy(idx) < bestEntropy)
       bestTheta = theta;
       bestEntropy = entropy(idx);
       bestProj = proj;
    end
    idx = idx + 1;
end

minBestProj = abs(min(bestProj));
bestProj = bestProj + minBestProj;
maxBestProj = max(bestProj);
bestProj = bestProj ./ maxBestProj;
bestProj = bestProj .* 255;
bestProj = reshape(bestProj, h, w);
intr = uint8(bestProj);
intr  = imresize(intr, [256 256]);
disp(bestTheta)

x2 = cos(bestTheta * pi / 180);
y2 = sin(bestTheta * pi / 180);
u2 = [x2; y2];
u2t = u2';
P_theta = mtimes(u2, u2t);
chi_theta = double(zeros(2,num));
parfor n = 1:num
    chi_theta(:,n)=P_theta * chromaticityVec(:,n);%chi_theta = chi.dot(P_theta), X_th = p_th*X
end
%3-D log ratio
rho_estim = double(zeros(3,num));
parfor n = 1:num
    rho_estim(:,n)=U' * chi_theta(:,n);          %rho_estim = chi_theta.dot(U)
end
mean_estim = exp(rho_estim);     
estim = double(zeros(h, w, 3));                  
estim1 = reshape(mean_estim(1,:,:),h,w);
estim2 = reshape(mean_estim(2,:,:),h,w);
estim3 = reshape(mean_estim(3,:,:),h,w);
sum=estim1+estim2+estim3;
estim(:,:,1) = estim1./sum;
estim(:,:,2) = estim2./sum;
estim(:,:,3) = estim3./sum;

% add energy of brightest pixels
xe = -sin(bestTheta * pi / 180);
ye = cos(bestTheta * pi / 180);
ue = [xe; ye];                                   %illumination vector
mX    = chromaticityVec'*ue;                     %illumination brightness
mX_th = chi_theta'*ue;                           %
[~,bidx_] = sort(I_gray_valid(:),'descend');     %get the indices for illumination brightness
bidx = bidx_(1:ceil(0.3 * h * w ));              %the 30% brightest pixels
X_E = (median(mX(bidx))-median(mX_th(bidx)))*ue; %extra illumination
X_th = bsxfun(@plus,chi_theta,X_E);              %add back extra illumination
chi_new=X_th;

%new 3-D log ratio
rho_estim_new = double(zeros(3,num));
for n = 1:num
    rho_estim_new(:,n)=U' * chi_new(:,n);  %rho_estim = chi_theta.dot(U)
end
mean_estim_new = exp(rho_estim_new);
estim_ = double(zeros(h, w, 3));
estim1_ = reshape(mean_estim_new(1,:,:),h,w);
estim2_ = reshape(mean_estim_new(2,:,:),h,w);
estim3_ = reshape(mean_estim_new(3,:,:),h,w);
sum_=estim1_+estim2_+estim3_;
estim_(:,:,1) = estim1_./sum_;
estim_(:,:,2) = estim2_./sum_;
estim_(:,:,3) = estim3_./sum_;
figure(666), clf, imshow([I_org, estim, estim_]);
