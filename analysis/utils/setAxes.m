function [ ax] = setAxes(f)
ax=f.CurrentAxes;
ax.FontSize=16;
ax.LabelFontSizeMultiplier=1.2;
ax.TitleFontSizeMultiplier = 1.2;
ax.FontSmoothing = 'on';
ax.XTickLabelRotation=0;
end

