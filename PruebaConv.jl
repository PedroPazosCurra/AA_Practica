using DelimitedFiles
using Statistics
using JSON
using Flux
using XLSX: readdata
using Random
using DataFrames
using ScikitLearn
using Flux.Losses

@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier


alcoholoneHotEncoding(vector,umbral::Number=5.5) = 
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


funcionTransferenciaCapasConvolucionales = relu;



function entrenaRNA(topology::AbstractArray{<:Int,1}, dataset::Tuple{Matrix{Vector}, AbstractArray{Bool,2}}, validacion::Tuple{Matrix{Vector},
	AbstractArray{Bool,2}}=(), test::Tuple{Matrix{Vector},
	AbstractArray{Bool,2}}=(), maxEpochsVal::Int64=20, maxEpochs::Int64=10000, minLoss::Real=0, learningRate::Real=0.001)

	# Ojo 1, cercionarse de que inputs y targets tengan cada patrón en cada columna. La transpongo con ' pero ver si falla.
	# Ojo 2, las matrices que se pasan para entrenar deben ser disjuntas a las que se usen para test.
	
	inputs = dataset[1]
	targets = dataset[2]
	lossVector = zeros(maxEpochs)

	test_set=[test]
	
	if (!isempty(validacion))
		inval = validacion[1]
		outval = validacion[2]
		lossVectorValidacion = zeros(maxEpochs)
	end
	# Creo RNA que vamos a entrenar 
	ann = Chain(
    Conv((1,1), 1=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),

    MaxPool((2,1)),
    Conv((1,1), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,1)),
    Conv((1,1), 32=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,1)),
    Dense(288, 1, σ)
)
	
loss(x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
accuracy(batch) = mean(onecold(ann(batch[1])) .== onecold(batch[2]));
opt = ADAM(0.001);

train_set=[inputs,targets];
accuracy(train_set) = mean(onecold(ann(train_set[1])) .== onecold(train_set[2]));

println("Comenzando entrenamiento...")
mejorPrecision = -Inf;
criterioFin = false;
numCiclo = 0;
numCicloUltimaMejora = 0;
mejorModelo = nothing;

while (!criterioFin)

    # Hay que declarar las variables globales que van a ser modificadas en el interior del bucle
    global numCicloUltimaMejora, numCiclo, mejorPrecision, mejorModelo, criterioFin;

    # Se entrena un ciclo
	loss(x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
    Flux.train!(loss, params(ann), train_set, opt);

    numCiclo += 1;*

    # Se calcula la precision en el conjunto de entrenamiento:
    precisionEntrenamiento = mean(accuracy.(train_set));
    println("Ciclo ", numCiclo, ": Precision en el conjunto de entrenamiento: ", 100*precisionEntrenamiento, " %");

    # Si se mejora la precision en el conjunto de entrenamiento, se calcula la de test y se guarda el modelo
    if (precisionEntrenamiento >= mejorPrecision)
        mejorPrecision = precisionEntrenamiento;
        precisionTest = accuracy(test_set);
        println("   Mejora en el conjunto de entrenamiento -> Precision en el conjunto de test: ", 100*precisionTest, " %");
        mejorModelo = deepcopy(ann);
        numCicloUltimaMejora = numCiclo;
    end

    # Si no se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje
    if (numCiclo - numCicloUltimaMejora >= 5) && (opt.eta > 1e-6)
        opt.eta /= 10.0
        println("   No se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje a ", opt.eta);
        numCicloUltimaMejora = numCiclo;
    end

    # Criterios de parada:

    # Si la precision en entrenamiento es lo suficientemente buena, se para el entrenamiento
    if (precisionEntrenamiento >= 0.900)
        println("   Se para el entenamiento por haber llegado a una precision de 99.9%")
        criterioFin = true;
    end

    # Si no se mejora la precision en el conjunto de entrenamiento durante 10 ciclos, se para el entrenamiento
    if (numCiclo - numCicloUltimaMejora >= 10)
        println("   Se para el entrenamiento por no haber mejorado la precision en el conjunto de entrenamiento durante 10 ciclos")
        criterioFin = true;
    end
end


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

calculateMinMaxNormalizationParameters(x::Matrix{Vector}) = 
	(minimum(x, dims=1), maximum(x, dims=1))


function minmaxvec(x::Matrix{Vector})
	min1=0
	min2=0
	min3=0
	max1=0
	max2=0
	max3=0
    for i in 1:264
        minaux=minimum(x[i,1])
        maxaux=maximum(x[i,1])
		min1=minimum([min1,minaux])
		max1=maximum([max1,maxaux])
        minaux=minimum(x[i,2])
        maxaux=maximum(x[i,2])
		min2=minimum([min2,minaux])
		max2=maximum([max2,maxaux])
        minaux=minimum(x[i,3])
        maxaux=maximum(x[i,3])
		min3=minimum([min3,minaux])
		max3=maximum([max3,maxaux])
    end	
    #return (min1,max1,min2,max2,min3,max3)
	return(minimum([min1,min2,min3]),maximum([max1,max2,max3]))
end

	

function normalizeMinMax!(x::Matrix{Vector}, y::Tuple{Float64, Float64})
	minim = y[:][1]
	maxim = y[:][2]
	for i in 1:264
		x[i,1] .= (x[i] .- minim) ./ (maxim .- minim) # Añadir caso en el que min y max sean iguales
		x[i,2] .= (x[i] .- minim) ./ (maxim .- minim)
		x[i,3] .= (x[i] .- minim) ./ (maxim .- minim)
	end	
end
	
normalizeMinMax!(x::Matrix{Vector}) =
	normalizeMinMax!(x,minmaxvec(x))
	
function normalizeMinMax(x::Matrix{Vector}, y::NTuple{2, AbstractArray{<:Real,2}})
	bar = copy(x)
	normalizeMinMax!(bar,y)
	end
	
normalizeMinMax(x::Matrix{Vector}) =
	normalizeMinMax!(copy(x))



funcionTransferenciaCapasConvolucionales = relu;


ann = Chain(
    Conv((1,1), 2=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),

    MaxPool((2,1)),
    Conv((1,1), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,1)),
    Conv((1,1), 32=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,1)),
    x -> reshape(x, :, size(x, 4)),
    Dense(288, 1),
    softmax
)

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

inputsMatrix = Array{Vector}(undef,264,3)
for i in 1:size(il,1)
	inputsMatrix[i,1] = il[i]
	inputsMatrix[i,2] = rl[i]
	inputsMatrix[i,3] = al[i]
end	

normalizeMinMax!(inputsMatrix)
outputsMatrix = alcoholoneHotEncoding(parse.(Float64,ol),5.5);
ilMatrix = Array{Vector}(undef,264,1)
for i in 1:264
	ilMatrix[i]=inputsMatrix[i,1]
end

reparto = holdOut(264,0.0,0.2)

ann = entrenaRNA([32, 64],(ilMatrix[reparto[1],:], outputsMatrix[reparto[1],:]), (ilMatrix[reparto[2],:], outputsMatrix[reparto[2],:]), (ilMatrix[reparto[3],:], outputsMatrix[reparto[3],:]))
