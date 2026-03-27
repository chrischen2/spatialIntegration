function [sct ] = shortCellType(cellType)
switch cellType
    case 'RGC\OFF-sustained'
        sct='OffS';
    case 'RGC\OFF-transient'
        sct='OffT';
    case 'RGC\ON-alpha'
        sct='OnS';
    case 'RGC\ON-transient'
        sct='OnT';
    case 'amacrine\AII'
        sct='A2';
    case 'bipolar\cone bipolar'
        sct='coneBP';
    case 'bipolar\rod bipolar'
        sct='rodBP';
    case 'RGC'
        sct='other RGCs';
    case 'amacrine'
        sct='ACs';
    case 'RGC\ON-parasol'
        sct='OnParasol';
    case 'RGC\OFF-parasol'
        sct='OffParasol';
    case 'RGC\ON-midget'
        sct='OnMidget';
    case 'RGC\OFF-midget'
        sct='OffMidget';
    otherwise
        sct=cellType;
        
end
end

