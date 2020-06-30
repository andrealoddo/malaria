function [ds] = merge_sets(varargin)
%merge_sets concatena più datastore preservando le labels

%Merge sets
for i=2:nargin
   
    imds_cat = imageDatastore(cat(1, varargin{1}.Files, varargin{i}.Files)); 
    imds_cat.Labels = cat(1, varargin{1}.Labels, varargin{i}.Labels);
    varargin{1} = imds_cat;
    
end

ds = shuffle(varargin{1});

end

