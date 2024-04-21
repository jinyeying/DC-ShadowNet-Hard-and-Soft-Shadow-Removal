function [total_dist_12, total_pixel_12, ...
          total_dist_14, total_pixel_14, ...
          total_dist_16, total_pixel_16] = evaluate_recovery(gt, recovered, mask_12, mask_14, mask_16)
      
    gt = double(gt)/255;
    recovered = double(recovered)/255;

    if sum(~isnan(mask_12(:)))
      if numel(mask_12) ~= numel(recovered)/3
        mask_12 = imresize(mask_12, [size(recovered,1) size(recovered,2)],'nearest');
      end
    else
      mask_12 = ones([size(recovered,1) size(recovered,2)]);
      %mask_12 = uint8(mask_12);
    end
    if sum(~isnan(mask_14(:)))
      if numel(mask_14) ~= numel(recovered)/3
        mask_14 = imresize(mask_14, [size(recovered,1) size(recovered,2)],'nearest');
      end
    else
      mask_14 = ones([size(recovered,1) size(recovered,2)]);
      %mask_14 = uint8(mask_14);
    end
    if sum(~isnan(mask_16(:)))
      if numel(mask_16) ~= numel(recovered)/3
        mask_16 = imresize(mask_16, [size(recovered,1) size(recovered,2)],'nearest');
      end
    else
      mask_16 = ones([size(recovered,1) size(recovered,2)]);
      %mask_16 = uint8(mask_16);
    end

    if numel(gt) ~= numel(recovered)
      gt = imresize(gt, [size(recovered,1) size(recovered,2)]);
    end
    
    %{
    Can we replace this 3-line code with the rgb2lab function mentioned
    here -
    https://pastebin.com/uuV8LFTi
    How to check?
    - Replace and see if the values of "recovered" varaible
      is exactly the same. If yes, the code can be replaced.
    %}
    cform = makecform('srgb2lab');
    gt = applycform(gt,cform);
    recovered = applycform(recovered,cform);

    dist = abs((gt - recovered).* repmat(mask_12,[1 1 3]));
    total_dist_12  = sum(dist(:));
    total_pixel_12 = sum(mask_12(:)); 
    
    dist = abs((gt - recovered).* repmat(mask_14,[1 1 3]));
    total_dist_14  = sum(dist(:));
    total_pixel_14 = sum(mask_14(:)); 
    
    dist = abs((gt - recovered).* repmat(mask_16,[1 1 3]));
    total_dist_16  = sum(dist(:));
    total_pixel_16 = sum(mask_16(:)); 
   
end
