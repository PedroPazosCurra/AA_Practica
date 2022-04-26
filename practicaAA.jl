using DelimitedFiles
using Statistics
using JSON
using Flux
using XLSX: readdata
using Random
using DataFrames
using ScikitLearn

alcoholoneHotEncoding(vector,umbral::Number=3.3) = 
	reshape(vector .< umbral,(size(vector,1),1))

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

# Se calcula la perdida (loss), que se utilizará para modificar los pesos de las conexiones (bias)
# x -> entradas			y -> salidas deseadas
loss(x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(ann(x), y) : Losses.crossentropy(ann(x), y);

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

function entrenaRNA(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, maxEpochs::Int64=1000, minLoss::Real=0, learningRate::Real=0.01)
	# Ojo 1, cercionarse de que inputs y targets tengan cada patrón en cada columna. La transpongo con ' pero ver si falla.
	# Ojo 2, las matrices que se pasan para entrenar deben ser disjuntas a las que se usen para test.
	
	inputs = dataset[1]
	targets = dataset[2]
	lossVector = zeros(maxEpochs)
	# Creo RNA que vamos a entrenar 
#	ann = creaRNA(topology, size(inputs,1), size(targets,1))

	ann = Chain(
		Dense(3,264,σ),
		Dense(264,1,σ)
	)
	
	loss(x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(ann(x), y) : Losses.crossentropy(ann(x), y);

	# Bucle para entrenar cada ciclo!!!
	aux = 0
	while ((loss(inputs',targets') > minLoss) && (aux < maxEpochs)) # Mientras el loss no sea ok(??) && no me pase de intentos max

		Flux.train!(loss, params(ann), [(inputs', targets')], ADAM(learningRate)); 

		lossVector[aux+1] = loss(inputs',targets')
		aux += 1
	end
	
	# Devuelvo RNA entrenada y un vector con los valores de loss en cada iteración.
	return (ann, lossVector)

end

function experimentoRNA(dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}})
	# Función para facilitar el experimento con la técnica de RNA. Voy a coger 8 arquitecturas a pelo con entre 1 y 2 ocultas.
	listaRNAs = []
	listaDicts = []
	dictAux = Dict()

	# Lista de topologías que vamos a meter a nuestras 8 RNA de experimentación.
	topologyArray = [[2], [4], [2,2], [2,3], [2,4], [3,3], [3,4], [4,4]]

	# Divido el dataset. Necesito datos de entrenamiento, validacion y de test
	datasetArray = crossvalidation(dataset[2], 3)

	# Luego cambiar esto con la funcion conjuntos de gabriel

	entrenaInputArray = AbstractArray{<:Real,2}
	validInputArray = AbstractArray{<:Real,2}
	testInputArray = AbstractArray{<:Real,2}

	entrenaOutputArray = AbstractArray{<:Real,2}
	validOutputArray = AbstractArray{<:Real,2}
	testOutputArray = AbstractArray{<:Real,2}

	for i in 1:size(datasetArray,1)
		if (datasetArray[i] == 1)
			append!(entrenaInputArray, [dataset[1][i, :]])
			append!(entrenaOutputArray, [dataset[2][i, :]])
		end
		if (datasetArray[i] == 2)
			append!(validInputArray, [dataset[1][i, :]])
			append!(validOutputArray, [dataset[2][i, :]])
		end
		if (datasetArray[i] == 3)
			append!(testInputArray, [dataset[1][i, :]])
			append!(testOutputArray, [dataset[2][i, :]])
		end
	end


	# Creo y entreno las 8 RNA y hago una lista con ellas
	for i in [1:8]

		append!(listaRNAs, entrenaRNA(topologyArray[i], (entrenaInputArray, entrenaOutputArray) )[0])
		
	end
	
	# Pruebo las 8 RNA para el conjunto de test y guardo su rendimiento (confussion matrix)
	for i in [1:8]
		
		outputsPrueba = classifyOutputs( reshape(listaRNA[i](testInputArray), (1,1)) )
	
		dictAux = confusionMatrix(outputsPrueba , reshape(listaRNA[i](testOutputArray), (1,1)))
		append!(listaDicts, dictAux)

	end

	# Devuelvo una lista con el rendimiento de los 8 modelos
	return listaDicts

end


#function testeame(modelo::<:Any, inputs::AbstractArray{<:Any,1} , targets::AbstractArray{<:Any,1})
	# Función que dado un modelo cualquiera, inputs y output esperado (target) saca la matriz de confusión.

	#outputsModelo = classifyOutputs(reshape(modelo(inputs), (1,1)))
	
	#return confussionMatrix(outputsModelo,targets)	
#end

function conjuntos(inputs::AbstractArray{<:Any,2},outputs::AbstractArray{<:Any,2},k::Int64)
	v = crossvalidation(outputs,k)
	testIArray= Array{<:Any,2}
	trainIArray= Array{<:Any,2}
	validIArray= Array{<:Any,2}
	testOArray= Array{<:Any,2}
	trainOArray= Array{<:Any,2}
	validOArray= Array{<:Any,2}
	

	for i in 1:size(v,1)
		if (v[i] == k)
			append!(testIArray, inputs[i, :])
			append!(testOArray, [outputs[i]])
		else
			if (v[i] == (k-1))
				append!(validIArray, inputs[i, :])
				append!(validOArray, [outputs[i]])
			else
				append!(trainIArray, inputs[i, :])
				append!(trainOArray,[outputs[i]])
			end 
		end	
	return Dict("entrenamiento" => (trainIArray,trainOArray), "test" => (testIArray,testOArray), "validacion" => (validIArray,validOArray))
	end
end


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




function modelCrossValidation(modelName::String, inputs::Array{<:Any,2},targets::Array{<:Any,1}, k::Int64, parameters::Dict, topology::AbstractArray{<:Int,1} = [1,1])

	if (modelName == "RNA")
		#model = entrenaRNA(topology,(inputs,oneHotEncoding(targets)))[1]
	else 
		if (modelName == "SVM")
			model = SVC(kernel=parameters["kernel"], degree=parameters["degree"], gamma=parameters["gamma"], C=parameters["C"])
		end
		if (modelName == "Árbol Decisión")
			model = DecisionTreeClassifier(max_depth=parameters["max_depth"], random_state=parameters["random_state"])
		end
		if (modelName == "kNN")
			model = KNeighborsClassifier(parameters["k"])		
		end
		fit!(model, inputs, targets)
	end	
	return model
end

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

function integrationRna(seed::Int64, input::AbstractArray{<:Any,1}, output, k::Int64)
	Random.seed!(seed)
	#input = oneHotEncoding(input)
	v = Vector{<:Any}(1:k)
	result = zeros(k)

	for i in 1:k
		d = conjuntos(input,output,k)
		v[i] = (entrenaRNA([1 2],d["entrenamiento"])[1], d["test"])
	end
	# (rna, (in, out))
	for i in 1:length(v)
		result[i] = accuracy(classifyOutputs(v[i][1](v[i][2][1])), v[i][2][2])
	end
	return mean(result)
end

 

function integration(seed::Int64, input::AbstractArray{<:Any,2}, output, k::Int64, parameters)
	Random.seed!(seed)

	d =	conjuntos(input,output,k)
	model = SVC(kernel=parameters["kernel"], degree=parameters["degree"], gamma=parameters["gamma"], C=parameters["C"])
	fit!(model, d["entrenamiento"][1], d["entrenamiento"][2])

	return (model, d["test"])
end

parameters = Dict();
parameters["kernel"] = "rbf";
parameters["kernelDegree"] = 3;
parameters["kernelGamma"] = 2;
parameters["C"] = 1;



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

dataset1 = JSON.parsefile("Cerveza.json")
dataset2 = JSON.parsefile("Cerveza2.json")
dataset = merge(dataset1,dataset2)
wl=[]
il=[]
al=[]
rl=[]
ol=[]
for d in values(dataset)
    push!(wl,get(d,"Wavelength",0)[52:114])
    push!(al,get(d,"Absorbance",0)[52:114])
    push!(rl,get(d,"Reflectance",0)[52:114])
    push!(il,get(d,"Intensity",0)[52:114])
    push!(ol,d["Labels"]["Graduacion"])
end
inputsMatrix = zeros(264,3)
for i in 1:size(il,1)
	inputsMatrix[i,1] = mean(il[i])
	inputsMatrix[i,2] = mean(rl[i])
	inputsMatrix[i,3] = mean(al[i])
end	

normalizeMinMax!(inputsMatrix)
outputsMatrix = alcoholoneHotEncoding(parse.(Float64,ol))


function getData(dict)
	matrix = zeros(228,176,3)
	next = iterate(dict)
	j = 1
	i = 1
	while next !== nothing
		(i,state) = next
		try
			matrix[:,j,1] = i[2]["Absorbance"]
			i = 1
			matrix[:,j,2] = i[2]["Reflectance"]
			i = 2
			matrix[:,j,3] = i[2]["Intensity"]
			i = 3
			j += 1
		catch KeyError
		end
		next = iterate(dict,state)
	end
	return matrix
end


dataset1 = JSON.parsefile("Cerveza.json")
dataset2 = JSON.parsefile("Cerveza2.json")
dataset = merge(dataset1,dataset2)

trainset = inputsMatrix[1:234, :]
testset = inputsMatrix[235:size(inputsMatrix,1), :]
trainout = outputsMatrix[1:234]
testout = outputsMatrix[235:size(inputsMatrix,1)]

model = KNeighborsClassifier(2); 

fit!(model, trainset, trainout)
expout = predict(model, testset); 
d = confusionMatrix(expout,vec(testout))

# VN FN ; FP VP




parameters = Dict("kernel" => "linear", "degree" => 3, "gamma" => 2, "C"=> 1)

#tupla1 = integration(1234, inputsMatrix, outputsMatrix, 10, parameters)
#testOutputs = predict(model, tupla1[2][1]); 
#confusionMatrix(testOutputs,tupla1[2][2])



# Aprox 3
dataset1 = JSON.parsefile("C:\\Users\\gabri\\OneDrive\\Escritorio\\AA\\Practica\\Cerveza.json")
dataset2 = JSON.parsefile("C:\\Users\\gabri\\OneDrive\\Escritorio\\AA\\Practica\\Cerveza2.json")
dataset = merge(dataset1,dataset2)

wl=[]
il=[]
al=[]
rl=[]
ol=[]
for d in values(dataset)
    push!(wl,get(d,"Wavelength",0)[52:114])
    push!(al,get(d,"Absorbance",0)[52:114])
    push!(rl,get(d,"Reflectance",0)[52:114])
    push!(il,get(d,"Intensity",0)[52:114])
    push!(ol,d["Labels"]["Graduacion"])
end
inputsMatrix = zeros(264,3)
for i in 1:size(il,1)
	inputsMatrix[i,1] = mean(il[i])
	inputsMatrix[i,2] = mean(rl[i])
	inputsMatrix[i,3] = mean(al[i])
end	

normalizeMinMax!(inputsMatrix)
outputsMatrix = alcoholoneHotEncoding(parse.(Float64,ol))

datasetaprox3 = [inputsMatrix[:, 1] (inputsMatrix[:, 2] .* inputsMatrix[:, 3])]

trainset = inputsMatrix[1:234, :]
testset = inputsMatrix[235:size(inputsMatrix,1), :]
trainout = outputsMatrix[1:234]
testout = outputsMatrix[235:size(inputsMatrix,1)]

model = KNeighborsClassifier(2);

fit!(model, trainset, trainout)
expout = predict(model, testset); 
d = confusionMatrix(expout,vec(testout))