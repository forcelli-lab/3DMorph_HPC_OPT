%% Parameter Formatter
%INPUT: Loads my parameter Excel Sheet and formats it correctly
%Excel sheet format by column, parameters generated by 3DMORPH Interactive Mode:
%A:FileName
%B: CellsizeCutoff
%C: SmCellSizeCutoff
%D: Scale
%E: zscale
%F: noise
%G: adjust
%OUTPUT: .mat file with parameters for 3DMORPH_HPC_OPT
%NK2024 Forcelli Lab

%% INSERT DATA
filename='13022024_GliaMorphologyParameters.xlsx';%INSERT EXCEL FILENAME HERE

BaseParameterFile='Parameters_Base.mat';
load(BaseParameterFile);
%% SCRIPT DATA
excelSheet=readcell(filename);

for i=83:numel(excelSheet(:,1))
    FileList(1,i-1)=strcat(string(excelSheet(i,1)),'.tif'); 
    adjust(1,i-1)=excelSheet(i,7);
    noise(1,i-1)=excelSheet(i,6);
    zscale(1,i-1)=excelSheet(i,5);
    scale(1,i-1)=excelSheet(i,4);
    SmCellCutoff(1,i-1)=excelSheet(i,3);
    CellSizeCutoff(1,i-1)=excelSheet(i,2);
end

savename="Parameters_HPC3DMORPH";
save(savename,'FileList','ch','ChannelOfInterest','scale','zscale','adjust','noise','s','ShowImg','ShowObjImg','ShowCells','ShowFullCells','CellSizeCutoff','SmCellCutoff','KeepAllCells','RemoveXY','ConvexCellsImage','SkelMethod','SkelImg','OrigCellImg','EndImg','BranchImg','BranchLengthFile');
