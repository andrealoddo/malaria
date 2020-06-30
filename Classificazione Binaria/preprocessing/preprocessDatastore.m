function [] = preprocessDatastore(ds)
%Svolge il preprocessing per ogni immagine del dataset e salva il risultato 

numRows = size(ds.Files,1);
 
%Per ogni immagine del dataset
for idx = 1:numRows
    
    imgOut = readimage(ds,idx);
    
    %Applica il preprocessing
    imgOut = preprocess_malaria_images(imgOut);
   
    %Salva il risultato
    imwrite(imgOut,ds.Files{idx});
end
end