function [] = crop_cells_from_image(immagine,gt,path)
% La funzione applica la maschera gt all'immagine, estrae le componenti e ne svolge il crop.
% Salva ogni crop come immagine componendo un datastore.


% Ricerca delle componenti connesse
[L,n] = bwlabel(gt,8);
x=0;

% Per ogni componente connessa
for i = 1:n
   
    componente = L==i;
    componente = im2uint8(componente);
    componente(componente==255)=1;
    
    % Si applica la maschera all'immagine ricavando il crop della cellula
    cellula = immagine .* componente;
    
    % Si estra il crop dal resto del background
    foregroundcol = any(componente);
    firstforegroundcol = find(foregroundcol, 1);
    lastforegroundcol = find(foregroundcol, 1, 'last');
    foregroundrow = any(componente,2);
    firstforegroundrow = find(foregroundrow, 1);
    lastforegroundrow = find(foregroundrow, 1, 'last');

    foregroundImg = cellula(firstforegroundrow:lastforegroundrow, firstforegroundcol:lastforegroundcol,:);
    
    %preprocessin
    %foregroundImg = preprocess_malaria_images(foregroundImg);
    
    
    
    % Salvataggio del crop in una specifica sottocartella
    class = path(end-4:end);
    class = class(1);
    
    s = 'C:\Users\Corrado\Desktop\Transfer Learning for Malaria Diagnosis\F\';
    
    switch class
        case 'R'
            imwrite(foregroundImg,[s 'r\' path(end-20:end-4) int2str(x) '.png']);

        case 'S'                        
            imwrite(foregroundImg,[s 's\' path(end-20:end-4) int2str(x) '.png']);

        case 'T'                        
            imwrite(foregroundImg,[s 't\' path(end-20:end-4) int2str(x) '.png']);

        case 'G'            
            imwrite(foregroundImg,[s 'g\' path(end-20:end-4) int2str(x) '.png']);
    end
    
    x=x+1;
end

