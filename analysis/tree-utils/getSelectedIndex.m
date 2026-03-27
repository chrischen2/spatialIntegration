function selectedIndex = getSelectedIndex(epochList)

  for epoch = 1:epochList.length
    isSelected(epoch) = epochList.valueByIndex(epoch).isSelected;
  end
selectedIndex = find(isSelected == 1);
 
end