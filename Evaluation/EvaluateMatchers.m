function [ SPs, APs , num] = EvaluateMatchers( ResultsDir, Matchers, AngleThreshold)

fileGt  = [ResultsDir 'groundtruth.txt'];

NumMethods = length(Matchers);
SPs = zeros(AngleThreshold, NumMethods);
APs = zeros(AngleThreshold, NumMethods);

for m = 1 : length(Matchers)     
    ResName = [ResultsDir Matchers{m} '.txt'];

    if(exist(ResName, 'file')==0)
        disp([ResName ' does not exist!']);
        continue;
    end
    [SP, AP, num] = ComparePoses(ResName, fileGt, AngleThreshold);
    SPs(:,m) = SP;
    APs(:,m) = AP;
end

end

function [ SP, AP, numbers ] = ComparePoses( fileRes,fileGt, AngleThreshold)
[ninliers, posesRes] = LoadResults(fileRes);
[posesGt] = LoadGt(fileGt);

num = size(ninliers,1);
SP = zeros(AngleThreshold,1);
AP = zeros(AngleThreshold,1);

for idx = 1 : num
    if(ninliers(idx) < 10)
        continue;
    end
         
    [rError, tError] = GetPoseError(posesRes(idx,:), posesGt(idx,:));
    Error = max(rError, tError);
    
    if(Error < AngleThreshold)
       SP(1 + floor(Error)) = SP(1 + floor(Error)) + 1;
       AP(1 + floor(Error)) = AP(1 + floor(Error)) + ninliers(idx);
    end
end

SP = cumsum(SP); 
AP = cumsum(AP);
numbers = num;

AP = AP ./ SP;
SP = SP / num; 

end

function [ ninliers, poses ] = LoadResults( fileRes )
A = dlmread(fileRes);
ninliers = A(:,3);
poses = A(:,4:15);
end

function [ posesGt ] = LoadGt( fileGt )
A = dlmread(fileGt);
posesGt = A(:,3:14);
end

function [rError, tError] = GetPoseError(poseRes, poseGt)
% pose is (1 * 12) matrix    
pose1 = reshape(poseRes,4,3);
pose1 = pose1';

pose2 = reshape(poseGt,4,3);
pose2 = pose2';

% Rotation
R1 = pose1(1:3, 1:3);
R2 = pose2(1:3, 1:3);
R_error = R1 \ R2;

a = R_error(1,1);
b = R_error(2,2);
c = R_error(3,3);
d = 0.5 * (a + b + c - 1.0);
rError = acos(max(min(d,1),-1)) * 180 / pi;
rError = abs(rError);

% Translation
t1 = pose1(1:3, 4); t1 = t1 / sqrt(t1'*t1);
t2 = pose2(1:3, 4); t2 = t2 / sqrt(t2'*t2);
tError = acos(t1' * t2) / pi * 180;
tError = abs(tError);

end
