% Script utilizzato per fare i crop delle immagini del dataset Falciparum

clear
clc

path1 = 'C:\Users\Corrado\Desktop\Transfer Learning for Malaria Diagnosis\Falciparum\img';
path2 = 'C:\Users\Corrado\Desktop\Transfer Learning for Malaria Diagnosis\Falciparum\gt';
ds1 = imageDatastore(path1);
ds2 = imageDatastore(path2);

numRows = size(ds1.Files,1);

% Per ogni immagine del dataset crea il crop delle cellule 
for idx = 1:numRows
    
    immagine = readimage(ds1,idx);
    gt = readimage(ds2,idx);
    path = ds1.Files{idx};
    crop_cells_from_image(immagine,gt,path);
end