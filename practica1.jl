using DelimitedFiles
using Statistics

dataset = readdlm("iris.data",',')
inputs = dataset[:,1:4];
targets = dataset[:,5];

inputs = convert(Array{Float32,2},inputs);
# Tambien inputs = Float32.(inputs), inputs = [Float32(x) for x in inputs]

function codificar(x)
	if (size(unique(x),1) == 2)
		x = unique(x)[1] .== x
		x = convert(Array{Float32,1},x)
	else
		x = unique(x) .== permutedims(x)
		x = convert(Array{Float32,2},x)	
	end
end

function valores(x)  
	print("Minimos: ", minimum(x, dims=1), "\n") 
	print("Maximos: ", maximum(x, dims=1), "\n")
	print("Media: ", mean(x, dims=1), "\n")
	print("Desviacion tipica: ", std(x, dims=1), "\n")
end
