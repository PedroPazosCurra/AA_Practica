using DelimitedFiles
using Statistics
using JSON
using Flux
using XLSX: readdata
using Random
using DataFrames
using ScikitLearn

Random.seed!(1234);

dataset1 = JSON.parsefile("Cerveza.json");
dataset2 = JSON.parsefile("Cerveza2.json");
dataset = merge(dataset1,dataset2);

wl=[];
il1=[];
il2=[];
il3=[];
il4=[];
al1=[];
al2=[];
al3=[];
al4=[];
rl1=[];
rl2=[];
rl3=[];
rl4=[];
ol=[];
range = 52:116;
range1 = 52:68;
range2 = 68:84;
range3 = 84:100;
range4 = 100:116;

for d in values(dataset)
    push!(wl,get(d,"Wavelength",0)[range]);
    push!(al1,get(d,"Absorbance",0)[range1]);
    push!(al2,get(d,"Absorbance",0)[range2]);
    push!(al3,get(d,"Absorbance",0)[range3]);
    push!(al4,get(d,"Absorbance",0)[range4]);
    push!(rl1,get(d,"Reflectance",0)[range1]);
    push!(rl2,get(d,"Reflectance",0)[range2]);
    push!(rl3,get(d,"Reflectance",0)[range3]);
    push!(rl4,get(d,"Reflectance",0)[range4]);
    push!(il1,get(d,"Intensity",0)[range1]);
    push!(il2,get(d,"Intensity",0)[range2]);
    push!(il3,get(d,"Intensity",0)[range3]);
    push!(il4,get(d,"Intensity",0)[range4]);
    push!(ol,d["Labels"]["Graduacion"]);
end

inputsMatrix = zeros(264,3);
for i in 1:size(il,1)
	inputsMatrix[i,1] = mean([maximum(il1[i]),maximum(il2[i]),maximum(il3[i]),maximum(il4[i])]);
	inputsMatrix[i,2] = mean([maximum(rl1[i]),maximum(rl2[i]),maximum(rl3[i]),maximum(rl4[i])]);
	inputsMatrix[i,3] = mean([maximum(al1[i]),maximum(al2[i]),maximum(al3[i]),maximum(al4[i])]);
end

normalizeMinMax!(inputsMatrix);
outputsMatrix = alcoholoneHotEncoding(parse.(Float64,ol),5.5);

inputsMatrix = [(inputsMatrix[:,1] .* inputsMatrix[:,2]) inputsMatrix[:,3]];

experimentoRNA((inputsMatrix,outputsMatrix));

@sk_import svm: SVC;
@sk_import tree: DecisionTreeClassifier;
@sk_import neighbors: KNeighborsClassifier;

experimentoSVC((inputsMatrix,outputsMatrix));
experimentoArboles((inputsMatrix,outputsMatrix));
experimentoKNN((inputsMatrix,outputsMatrix));
