clear; clc; close all;

Matchers = {'sift', 'surf', 'orb', 'akaze', 'brisk', 'kaze',  'dlco', 'freak', 'binboost', 'latch', 'daisy', 'star', 'msd',  'gms', 'gms_s' 'ransac_fm', 'usac_fm', 'lmeds_fm', 'lmeds_em' };
AngleThreshold = 15;
NumMethods = length(Matchers);

cdir = pwd;
ResultsDir = {[cdir '/../Results/01-office/'], ...
       [cdir '/../Results/02-teddy/'],...
       [cdir '/../Results/03-large-cabinet/'],...
       [cdir '/../Results/04-kitti/'],...
       [cdir '/../Results/05-castle/'],...
       [cdir '/../Results/06-office-wide/'], ...
       [cdir '/../Results/07-teddy-wide/'],...
       [cdir '/../Results/08-large-cabinet-wide/']
};

SPS = zeros(NumMethods, 16);

for idx = 1 : length(ResultsDir)
    [SP,  AP, numbers] = EvaluateMatchers( ResultsDir{idx}, Matchers, AngleThreshold);
    
    SPS(:,idx) = SP(5,:)';
    SPS(:,8+idx) = SP(15,:)';
end

[S, I] = sort(SPS,'descend');

fid = fopen('.\Tables\sp_table.txt', 'w');
fprintf(fid, '\\renewcommand{\\arraystretch}{1.3}\n');
fprintf(fid, '\\begin{table*}\n');
fprintf(fid, '\\centering\n');
fprintf(fid, '\\scriptsize\n');
fprintf(fid, '\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}\n');
fprintf(fid, '\\hline\n');
fprintf(fid, '& \\multicolumn{8}{c|}{pose error threshold: 5 degrees} &');
fprintf(fid, '\\multicolumn{8}{c|}{pose error threshold: 15 degrees}\\\\\n');
fprintf(fid, '\\hline\n');
fprintf(fid, 'Methods & 01 & 02 & 03 & 04 & 05 & 06 & 07 & 08 & 01 & 02 & 03 & 04 & 05 & 06 & 07 & 08\\\\\n');
fprintf(fid, '\\hline\n');
for row = 1 : NumMethods
    fprintf(fid,'%s  ', strrep(upper(Matchers{row}),'_','\_'));
    for col = 1 : 16 
        s = sprintf('%.3f', SPS(row,col));
        
        r1 = I(1,col); r2 = I(2,col); r3 = I(3,col);
        if(row == r1)
            s = sprintf('\\textcolor{red}{%s}',s);
        end
        if(row == r2)
            s = sprintf('\\textcolor{green}{%s}',s);
        end
        if(row == r3)
            s = sprintf('\\textcolor{blue}{%s}',s);
        end                      
        
        if(SPS(row,col) > SPS(1,col))
            s = sprintf('\\textbf{%s}',s);
        end
        
        if(SPS(row,col)==0)
            fprintf(fid,'& / ');
        else
            fprintf(fid,'& %s', s);
        end
    end
    fprintf(fid,'\\\\\n\\hline\n');
end

fprintf(fid, '\\end{tabular}\n');
fprintf(fid, '\\label{tab_matchbench_sp}\n');
fprintf(fid, '\\end{table*}\n');
fclose(fid);