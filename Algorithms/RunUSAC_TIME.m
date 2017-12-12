clear; clc;clear; clc; close all;

cdir = pwd;

Dataset = [cdir '/../Dataset/01-office/'];

cdir = pwd;
exe = [cdir '/bin/Release/usac_matcher_time'];
usac_dir = [cdir '/USAC/'];

cmd = [exe ' ' usac_dir ' ' Dataset];
system(cmd);
