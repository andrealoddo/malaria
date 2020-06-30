function [numCorrect] = num_pred_corr(testpreds,truetest)
% Questa funzione conta il numero di predizioni corrette: essendo tutte le
% cellule del test set infette, se la rete ha predetto "Uninfected" ha
% svolto una predizione errata

for i = 1:length(truetest)


    if testpreds(i)== {'Uninfected'}
        
        corr_vector(i)=0;    
        
    else
        corr_vector(i)=1;
    end


end

numCorrect = nnz(corr_vector);



