clear; clc; close all;
addpath('./export_fig');

Matchers = {'sift', 'surf', 'orb', 'akaze', 'brisk', 'kaze',  'dlco', 'freak', 'binboost', 'latch', 'daisy', 'star', 'msd',  'gms', 'gms_s' 'ransac_fm', 'usac_fm', 'lmeds_fm', 'lmeds_em' };
Legends = strrep(Matchers,'_','\_');
Colors = {'y', 'g', 'b', 'k', 'm', 'c', 'k-o', 'm-o', 'b-o', 'g-o', 'c-o', 'g-*', 'k-*', 'r-+', 'r-s', 'g--', 'b--', 'k--', 'm--'};
AngleThreshold = 15;
NumMethods = length(Matchers);

cdir = pwd;
ResultsDir = [cdir '/../Results/07-teddy-wide/'];

[SP,  AP] = EvaluateMatchers( ResultsDir, Matchers, AngleThreshold);

x = 1 : AngleThreshold;
h = figure;
for m = 1 : length(Matchers)
    plot(x, SP(:,m), Colors{m}, 'linewidth', 2);
    hold on;
end
legend(Legends,'Location','BestOutside', 'FontSize', 14);
grid on;
xlabel('Pose Error Threshold [Degrees]');

h = figure;
bar(AP(5,:)');
grid on;
xlabel('Pose Error Threshold [Degrees]');

