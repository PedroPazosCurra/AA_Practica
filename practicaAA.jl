using DelimitedFiles
using Statistics
using JSON
using Flux.Losses

oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}) = 
	if (size(classes,1) == 2)
		boolVector = feature .== classes[1]
		# Transformar boolVector a matriz bidimensional de una columna
	else
		
	end
	
oneHotEncoding(feature::AbstractArray{<:Any,1}) = 
	oneHotEncoding(feature,unique(feature))
	
oneHotEncoding(feature::AbstractArray{Bool,1}) = 
	boolVector = reshape(feature, (length(feature),1))
	
calculateMinMaxNormalizationParameters(x::AbstractArray{<:Real,2}) = 
	(minimum(x, dims=1), maximum(x, dims=1))
	
calculateZeroMeanNormalizationParameters(x::AbstractArray{<:Real,2}) = 
	(mean(x, dims=1), std(x, dims=1))
	
function normalizeMinMax!(x::AbstractArray{<:Real,2}, y::NTuple{2, AbstractArray{<:Real,2}})
	minim = y[:][1]
	maxim = y[:][2]
	x .= (x .- minim) ./ (maxim .- minim) # Añadir caso en el que min y max sean iguales
	end
	
normalizeMinMax!(x::AbstractArray{<:Real,2}) =
	normalizeMinMax!(x,calculateMinMaxNormalizationParameters(x))
	
function normalizeMinMax(x::AbstractArray{<:Real,2}, y::NTuple{2, AbstractArray{<:Real,2}})
	bar = copy(x)
	normalizeMinMax!(bar,y)
	end
	
normalizeMinMax(x::AbstractArray{<:Real,2}) =
	normalizeMinMax!(copy(x))

function normalizeZeroMean!(x::AbstractArray{<:Real,2}, y::NTuple{2, AbstractArray{<:Real,2}})
	media = y[:][1]
	desviacion = y[:][2]
	x .= (x .- media) ./ (media .- desviacion) # Añadir caso en el que desviacion tipica es 0
	end
	
normalizeZeroMean!(x::AbstractArray{<:Real,2}) =
	normalizeMinMax!(x,calculateZeroMeanNormalizationParameters(x))
	
function normalizeZeroMean(x::AbstractArray{<:Real,2}, y::NTuple{2, AbstractArray{<:Real,2}})
	bar = copy(x)
	normalizeZeroMean!(bar,y)
	end
	
normalizeZeroMean(x::AbstractArray{<:Real,2}) =
	normalizeZeroMean!(copy(x))
