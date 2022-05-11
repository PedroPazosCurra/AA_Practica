using DelimitedFiles
using Statistics
using JSON
using Flux
using XLSX: readdata
using Random
using DataFrames
using ScikitLearn
using Flux.Losses



###########################
# Introduccion
###########################



oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}) = 
	if (size(classes,1) == 2)
		boolVector = feature .== classes
		# Transformar boolVector a matriz bidimensional de una columna
        reshape(boolVector, (2,1))
	else
		boolMatrix =  BitArray{2}(0, size(feature,1), size(classes,1))
        for i in 1:size(classes,1)
            boolMatrix[:,i] = feature .== classes[i]
        end
	end
	
oneHotEncoding(feature::AbstractArray{<:Any,1}) = 
	oneHotEncoding(feature, unique(feature))
	
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

function classifyOutputs(x::AbstractArray{<:Real,2},threshold::Real=0.5)
	if (size(x,2) == 1)
		return x .>= threshold
	else
		(_,indicesMaxEachInstance) = findmax(x, dims=2)
		aux = falses(size(x))
		aux[indicesMaxEachInstance] .= true
		return aux
	end
end

function accuracy(targets::AbstractArray{Bool,1}, outputs::AbstractArray{Bool, 1})
    acc = targets .== outputs
    (sum(acc) * 100)/(length(acc))
    end

function accuracy(targets::AbstractArray{Bool,2}, outputs::AbstractArray{Bool, 2})
    if (size(targets,2) == 1) 
    	accuracy(targets[:,1],outputs[:,1])
    elseif (size(targets,2) > 2)
    	acc = 0
    	aux = targets .== outputs
    	for i in 1:size(aux,1)
    		if (sum(aux[i,:]) == size(targets,2))
    			acc += 1
    		end
    	end
    	acc = acc / size(targets,1)
    	return acc
    end
end

function accuracy(targets::AbstractArray{Bool,1},outputs::AbstractArray{<:Real, 1}, threshold::Real=0.5)
    aux = outputs .>= threshold
    accuracy(targets,aux)
end


function accuracy(targets::AbstractArray{Bool,2},outputs::AbstractArray{<:Real, 2})
    if (size(targets,2) == 1) 
    	accuracy(targets[:,1],outputs[:,1])
    elseif (size(targets,2) > 2)
    	aux = classifyOutputs(outputs)
    	accuracy(targets,aux)
    end
end


###########################
# RR NN AA
###########################



function creaRNA(topology::AbstractArray{<:Int,1}, numInputsLayer::Int64, numOutputsLayer::Int64)
	# RNA vacía
	ann = Chain();

	# Si hay capas ocultas, se itera por topology y se crea una capa por iteración
	for numOutputsLayer = topology 
		ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, σ) ); 
		numInputsLayer = numOutputsLayer; 
	end;

	# Devuelve rna creada!!
	return ann
end

function entrenaRNA(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, validacion::Tuple{AbstractArray{<:Real,2},
	AbstractArray{Bool,2}}=(), test::Tuple{AbstractArray{<:Real,2},
	AbstractArray{Bool,2}}=(), maxEpochsVal::Int64=20, maxEpochs::Int64=10000, minLoss::Real=0, learningRate::Real=0.001)

	# Ojo 1, cercionarse de que inputs y targets tengan cada patrón en cada columna. La transpongo con ' pero ver si falla.
	# Ojo 2, las matrices que se pasan para entrenar deben ser disjuntas a las que se usen para test.
	
	inputs = dataset[1]
	targets = dataset[2]
	lossVector = zeros(maxEpochs)
	
	if (!isempty(validacion))
		inval = validacion[1]
		outval = validacion[2]
		lossVectorValidacion = zeros(maxEpochs)
	end
	
	# Creo RNA que vamos a entrenar 
#	ann = creaRNA(topology, size(inputs,1), size(targets,1))

	ann = Chain(
		Dense(3,size(dataset,1),σ),
		Dense(size(dataset,1),1,σ),
	)
	
	loss(x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(ann(x), y) : Losses.crossentropy(ann(x), y);

	# Bucle para entrenar cada ciclo!!!
	aux = 1
	ctr = 0
	auxAnn = ann

	while ((loss(inputs',targets') > minLoss) && (aux < maxEpochs) && (ctr < maxEpochsVal)) 

		Flux.train!(loss, params(ann), [(inputs', targets')], ADAM(learningRate)); 

		lossVector[aux+1] = loss(inputs',targets')
		
		if (!isempty(validacion))
			lossVectorValidacion[aux+1] = loss(inval',outval')
			if (lossVectorValidacion[aux+1] >= lossVectorValidacion[aux])
				ctr += 1
			else
				ctr = 0
				auxAnn = ann
			end
		else 
			auxAnn = ann
		end

		aux += 1
	end
	
	# Devuelvo RNA entrenada y un vector con los valores de loss en cada iteración.
	# Si se da conjunto de validación, devuelve la rna con menor error de validación.
	return (auxAnn, lossVector)

end



###########################
# Sobreentrenamiento
###########################



function holdOut(N::Int64, P::Float64)
	v = randperm(N)
	cut = round(Int64,size(v,1)*P)
	return (v[1:cut], v[(cut + 1):(size(v,1))])
end

function holdOut(N::Int64, Pval::Float64, Ptest::Float64)
	t1 = holdOut(N, Pval + Ptest)
	t2 = holdOut(size(t1[2],1), Ptest)
	w1 = zeros(Int64,size(t2[1],1))
	w2 = zeros(Int64,size(t2[2],1))
	j = 1
	for i in t2[1]
		w1[j] = t1[2][i]
		j += 1
	end
	j = 1
	for i in t2[2]
		w2[j] = t1[2][i]
		j += 1
	end
	return (t1[1], w1, w2)

end


###########################
# Metricas
###########################



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
    if (vn != length(outputs))
	    sensibilidad = 	vp / (fn + vp)
        if(isnan(sensibilidad)) 
            sensibilidad = 0
        end
    end 

	especificidad = 	vn / (fp + vn) 
    if (isnan(especificidad)) 
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
	if (vp != length(outputs))
		v_pred_neg = 	vn / (vn + fn)
        if(isnan(v_pred_neg)) 
            v_pred_neg = 0
        end
	end

	f1_score = 0
	if (sensibilidad != 0 && v_pred_pos != 0)
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




###########################
# Clasificacion Multiclase
###########################

#  Dado que nuestra práctica pretende clasificar entre tan solo 2 clases, no se ha 
# hecho uso de estas funciones.

function multiClass(inputs::AbstractArray{<:Real, 2}, targets::AbstractArray{Bool,2})
	
	numClasses = size(inputs,2)
	numInstances = size(inputs,1)
	outputs = Array{Float32,2}(undef, numInstances, numClasses)

	for numClass in 1:numClasses
		model = fit(inputs, targets[:,[numClass]]);
		outputs[:,numClass] .= model(inputs);
	end

	outputs = softmax(outputs')'
	vmax = maximum(outputs, dims=2)
	# TODO :	"Esta última línea puede presentar problemas en caso de que
	# varios modelos generen la misma salida, ¿dónde estaría el
	# problema? ¿cómo se solucionaría?"
	outputs = (outputs .== vmax)
end


function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}, estrategia::String)

	@assert ((size(outputs,2) == size(targets,2)) && size(outputs != 2))
	@assert (estrategia == "macro" || estrategia == "weighted")

	sensibArr = zeros(size(outputs,2))
	especifArr = zeros(size(outputs,2))
	vppArr = zeros(size(outputs,2))
	vpnbArr = zeros(size(outputs,2))
	f1Arr = zeros(size(outputs,2))

	for i in size(outputs,2)
		confMatArr = confusionMatrix(outputs[i],targets[i])
		sensibArr[i] = confMatArr["sensibilidad"]
		especifArr[i] = confMatArr["especificidad"]
		vppArr[i] = confMatArr["valor_predictivo_positivo"]
		vpnbArr[i] = confMatArr["valor_predictivo_negativo"]
		f1Arr[i] = confMatArr["f1_score"]
	end
	
	if (estrategia == "macro")
		sensibilidad = mean(sensibArr) / size(sensibArr)
		especificidad = mean(especifArr) / size(especifArr)
		v_pred_pos = mean(vppArr) / size(vppArr)
		v_pred_neg = mean(vpnbArr) / size(vpnbArr)
		f1_score = mean(f1Arr) / size(f1Arr)
	else
		sumArr = zeros(Int64, size(targets,2))
		for j in size(targets,2)
			sumArr[j] = sum(targets[:,j])
		end

		for k in size(sumArr)
			sensibArr[k] *= sumArr[k]
			especifArr[k] *= sumArr[k] 
			vppArr[k] *= sumArr[k] 
			vpnbArr[k] *= sumArr[k] 
			f1Arr[k] *= sumArr[k] 
		end
		sensibilidad = mean(sensibArr) / size(sensibArr)
		especificidad = mean(especifArr) / size(especifArr)
		v_pred_pos = mean(vppArr) / size(vppArr)
		v_pred_neg = mean(vpnbArr) / size(vpnbArr)
		f1_score = mean(f1Arr) / size(f1Arr)
	end


	return Dict("valor_precision" => accuracy, "sensibilidad" => sensibilidad , "especificidad" => especificidad, "valor_predictivo_positivo" => v_pred_pos, "valor_predictivo_negativo" => v_pred_neg, "f1_score" => f1_score)
end

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}, estrategia::String)
	confusionMatrix(classifyOutputs(outputs), targets, estrategia)
end

function confusionMatrix(outputs::AbstractArray{<:Any}, targets::AbstractArray{<:Any}, estrategia::String)

	@assert(all([in(output, unique(targets)) for output in outputs]))
	uout = unique(outputs)
	utarg = unique(targets)
	uout = oneHotEncoding(uout)
	utarg = oneHotEncoding(utarg)
	confusionMatrix(uout,utarg,estrategia)
end


###########################
# Validacion Cruzada
###########################



# N > k
function crossvalidation(N::Int64, k::Int64)
	v = Vector{Int64}(1:k)
	vect = repeat(v,N)
	shuffle!(vect[1:N])
end

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
	v = Vector{Int64}(1:size(targets,1))
	for col in eachcol(targets)
		v = crossvalidation(sum(col),k)
	end
	return v
end

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
	crossvalidation(oneHotEncoding(targets),k)
end
# importante asegurarse de que se tienen al menos 10 patrones de cada clase

