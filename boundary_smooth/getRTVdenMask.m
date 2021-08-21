% Demo script
clearvars;
close all;

% Add RTV function library here
addpath(genpath('./RTV/'));

% Read the image here
input = (imread(['./122.png'])); 
I     = (imread(['./122_mask.png']));
if(size(I, 3)==1)
    I = cat3(I);
end

% Generate two different edge maps % Map2 is more smoothened than Map1
param1 = 0.01; 
param2 = 0.02;  
[~, Map1, Map2] = tsmooth(I,param1,param2);
Map1 = uint8(255*cat3(norm(Map1)));
Map2 = uint8(255*cat3(norm(Map2)));

figure(1), imshow([input, I, Map1]);
imwrite([input, I, Map1], ['./input_mask_boundary.jpg']);

function out = cat3(in)
out = cat(3, in, in, in);
end

function out= norm(in)
out = zeros(size(in));
for ch=1:size(in, 3)
    in_ch       = in(:, :, ch);
    in_ch_max   = max(in_ch(:));
    in_ch_min   = min(in_ch(:));
    in_ch_norm  = (in_ch - in_ch_min)./(in_ch_max - in_ch_min);
    out(:,:,ch) = in_ch_norm;
end
end

