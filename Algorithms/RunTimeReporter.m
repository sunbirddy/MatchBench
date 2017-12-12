clear; clc; close all;
cdir = pwd;
Dataset = [cdir '/../Dataset/01-office/'];

Matchers = {'sift', 'surf', 'orb', 'akaze', 'brisk', 'kaze', 'dlco','freak','binboost','latch','daisy','star', 'msd', 'gms', 'gms_s'};
%exe = [cdir '/bin/Release/time_reporter '];
exe = [cdir '/bin/Release/time_reporter_gpu '];


for idx = 1 : length(Matchers)
   disp(Matchers{idx});
   command = [exe Matchers{idx} ' ' Dataset];
   system(command);
    
end