function [] = preprocess_ds(ds)
 
numRows = size(ds.Files,1);
 
for idx = 1:numRows
    
    imgOut = readimage(ds,idx);
    
    imgOut = preprocess_malaria_images(imgOut);
   
    imwrite(imgOut,ds.Files{idx});
end
end