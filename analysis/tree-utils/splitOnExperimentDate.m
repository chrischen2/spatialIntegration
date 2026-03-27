function V = splitOnExperimentDate( epoch )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
   t=epoch.startDate; 
   V = datestr(t','yyyy/mm/dd');

end

