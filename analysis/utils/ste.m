function [ stError ] = ste( inputArr )
stError=std(inputArr)/sqrt(numel(inputArr)-1);
end

