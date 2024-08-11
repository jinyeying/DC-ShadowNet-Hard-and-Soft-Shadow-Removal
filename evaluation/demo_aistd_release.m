clear;
tic
fd = fopen('AISTDtest.txt');
a=textscan(fd, '%s');
fclose(fd);
testfnlist = a{1};

fprintf('Starting evaluation. Total %d images\n', numel(testfnlist));

total_dist_l2 = zeros(1, numel(testfnlist));
total_dist_l4 = zeros(1, numel(testfnlist));
total_dist_l6 = zeros(1, numel(testfnlist));
total_pix_l2 = zeros(1, numel(testfnlist));
total_pix_l4 = zeros(1, numel(testfnlist));
total_pix_l6 = zeros(1, numel(testfnlist));

parfor recovery_count = 1 : numel(testfnlist)    
    gt_recovery         = imread(['D:\Dropbox\0pytorch_project_6\dataset\AISTD\testB\' testfnlist{recovery_count}(1:end-4) '.png']);
    shadow_recovery     = imread(['D:\Dropbox\0pytorch_project_6\dataset\AISTD\testA\' testfnlist{recovery_count}]);    
 
    recovered_recovery  = imread(['D:\Dropbox\shadow_results\ICCV2021\DC-ShadowNet_RESULTS\DC-ShadowNet_AISTD\' testfnlist{recovery_count}(1:end-4) '.png']); 
    %recovered_recovery  = imread(['D:\Dropbox\shadow_results\ICCV2021\DC-ShadowNet_RESULTS\DC-ShadowNet_ISTD\' testfnlist{recovery_count}(1:end-4) '.png']); 
     
    m                   = imread(['D:\Dropbox\0pytorch_project_6\dataset\AISTD\testM\' testfnlist{recovery_count}(1:end-4) '.png']);

    if numel(size(m)) == 3
        m = rgb2gray(m);
    end

    m(m~=0)=1;

    m = double(m);

    mask_recovery = m;

    mask2_recovery = 1-m;

    % for the overall regions
    [total_dist_l2(1, recovery_count), ...
     total_pix_l2(1, recovery_count), ...
     total_dist_l4(1, recovery_count), ...
     total_pix_l4(1, recovery_count), ...
     total_dist_l6(1, recovery_count), ...
     total_pix_l6(1, recovery_count)] = evaluate_recovery(gt_recovery, ...
                                                          recovered_recovery, ...
                                                          NaN*ones(size(gt_recovery)),...
                                                          mask_recovery, ...
                                                          mask2_recovery);

end

dist_12 = sum(total_dist_l2(:))/sum(total_pix_l2(:));
dist_14 = sum(total_dist_l4(:))/sum(total_pix_l4(:));
dist_16 = sum(total_dist_l6(:))/sum(total_pix_l6(:));

[dist_14 dist_16 dist_12] %s, ns, overall
fprintf('%s/%.2f/%s/%.2f/%s/%.2f\n', 'Overall', dist_12, 'S', dist_14, 'NS', dist_16);     
%for the shadow evaluation, non_shadow, overall
fprintf('Evaluation complete! Total %d images in %.2f mins\n', numel(testfnlist), toc/60);

%AISTD,Overall/4.74/S/10.55/NS/3.65
%ISTD, Overall/7.77/S/12.77/NS/6.84
