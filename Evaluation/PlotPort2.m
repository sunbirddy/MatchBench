clear; clc; close all;
addpath('./export_fig');

Matchers = {'sift', 'surf', 'orb', 'akaze', 'brisk', 'kaze',  'dlco', 'freak', 'binboost', 'latch', 'daisy', 'star', 'msd',  'gms', 'gms_s' 'ransac_fm', 'usac_fm', 'lmeds_fm', 'lmeds_em' };
Legends = strrep(Matchers,'_','\_');
Colors = {'y', 'g', 'b', 'k', 'm', 'c', 'k-o', 'm-o', 'b-o', 'g-o', 'c-o', 'g-*', 'k-*', 'r-+', 'r-s', 'g--', 'b--', 'k--', 'm--'};
AngleThreshold = 15;
NumMethods = length(Matchers);

cdir = pwd;
ResultsDir = {[cdir '/../Results/05-castle/'],...
       [cdir '/../Results/06-office-wide/'], ...
       [cdir '/../Results/07-teddy-wide/'],...
       [cdir '/../Results/08-large-cabinet-wide/']};
   
titles = {'05-castle', '06-office-wide', '07-teddy-wide', '08-large-cabinet-wide'};

APS = zeros(4, NumMethods);
   
for idx = 1 : length(ResultsDir)
    [SP,  AP] = EvaluateMatchers( ResultsDir{idx}, Matchers, AngleThreshold);
    
    CurveName = ['./Figures/0' num2str(idx + 4)];

    x = 1 : AngleThreshold;
    h = figure;
    for m = 1 : length(Matchers)
        plot(x, SP(:,m), Colors{m}, 'linewidth', 2);      
        hold on;
    end
    if (idx == 1) 
        legend(Legends,'Location','BestOutside', 'FontSize', 10.0);
    end
    grid on;
    xlabel('pose error threshold [degrees]');
    ylabel('success ratio');
    title(titles{idx});
    %export_fig(h, [CurveName 'sp.pdf'], '-transparent');

    APS(idx,:) = AP(5,:)';
end

h = figure;
bar(APS);
grid on;
xlabel('pose error threshold: 5^o');
ylabel('number of correspondences');
legend(Legends,'Location','BestOutside', 'FontSize', 14);