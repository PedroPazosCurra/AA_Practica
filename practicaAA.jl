using DelimitedFiles
using Statistics
using JSON
using Flux.Losses

oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}) = 
	if (size(classes,1) == 2)
		boolVector = feature .== classes
		# Transformar boolVector a matriz bidimensional de una columna
        reshape(boolVector, (2,1))
	else
		boolMatrix =  BitArray{2}(0, size(feature,1), size(classes,1))
        for i in size(classes,1)
            boolMatrix[:,i] = feature .== classes[i]
        end
	end
	
oneHotEncoding(feature::AbstractArray{<:Any,1}) = 
	oneHotEncoding(feature,unique(feature))
	
oneHotEncoding(feature::AbstractArray{Bool,1}) = 
	reshape(feature, (length(feature),1))
	
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


function accuracy(targets::AbstractArray{Bool,1}, outputs::AbstractArray{Bool, 1})
    acc = targets .== outputs
    (sum(acc) * 100)/(length(acc))
    end


function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
	#Verdadero negativo:
	v1 = 0 .== outputs
	v2 = 0 .== targets
	aux = (v1 .&& v2)	
	vn = count(aux)
	
	#Verdadero positivo:		
	v1 = 1 .== outputs
	v2 = 1 .== targets
	aux = v1 .&& v2
	vp = count(aux)
	
	#Falso negativo:
	v1 = 0 .== outputs
	v2 = 1 .== targets
	aux = v1 .&& v2
	fn = count(aux)
	
	#Falso positivo:
	v1 = 1 .== outputs
	v2 = 0 .== targets
	aux = v1 .&& v2
	fp = count(aux)

    matr = [vn fp; fn vp]

	accuracy = 		(vn+vp)/(vn+fn+vp+fp)
    if(isnan(accuracy)) 
        accuracy = 0
    end
	tasa_fallo = 	(fn + fp)/(vn + vp + fn + fp)
    if(isnan(tasa_fallo)) 
        tasa_fallo = 0
    end

    sensibilidad = 1
    if(vn != length(outputs))
	    sensibilidad = 	vp / (fn + vp)
        if(isnan(sensibilidad)) 
            sensibilidad = 0
        end
    end 

	especificidad = 	vn / (fp + vn) 
    if(isnan(especificidad)) 
        especificidad = 0
    end

    v_pred_pos = 1
    if(vn != length(outputs))
		v_pred_pos = 	vp / (vp + fp)
        if(isnan(v_pred_pos)) 
            v_pred_pos = 0
        end
	end

	v_pred_neg = 1
	if(vp != length(outputs))
		v_pred_neg = 	vn / (vn + fn)
        if(isnan(v_pred_neg)) 
            v_pred_neg = 0
        end
	end

	f1_score = 0
	if(sensibilidad != 0 && v_pred_pos != 0)
		f1_score = 2 * (sensibilidad*v_pred_pos) / (sensibilidad+v_pred_pos)
        if(isnan(f1_score)) 
            f1_score = 0
        end
	end

    return Dict("valor_precision" => accuracy, "tasa_fallo" => tasa_fallo, "sensibilidad" => sensibilidad , "especificidad" => especificidad, "valor_predictivo_positivo" => v_pred_pos, "valor_predictivo_negativo" => v_pred_neg, "f1_score" => f1_score, "matriz_confusion" => matr)
end

function confusionMatrix(outputs::AbstractArray{<:Real}, targets::AbstractArray{Bool,1}, umbral::Number=0.5)
	confusionMatrix((outputs .>= umbral),targets)
end


Pkg.add("ScikitLearn")

@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier
# modelCrossValidation(tipoModelo::sabeDios, inputs::, outputs:: Array{Any,1}, k_cross_validation, parameters::Dict)
#   if tipoModelo RNA -> oneHotEncoding  Array{Any,1} y codificacion salida 
#       como en otras practicas
#
#   else confusionMatrix  Array{Any,1}
#       switch (tipoModelo)
#           case svm:  model = SVC(kernel=parameters["kernel"], degree=parameters["kernelDegree"], gamma=parameters["kernelGamma"], C=parameters["C"]);
#           case decisionTree:  model = DecisionTreeClassifier(max_depth=4, random_state=1) 
#           case knn:  KNeighborsClassifier(3); 
#       
#       fit!(model, trainingInputs, trainingTargets); 
#