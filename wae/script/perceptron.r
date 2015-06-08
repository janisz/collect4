if (!require("neuralnet")) {
  install.packages('neuralnet', repos="http://cran.rstudio.com/")
  library(neuralnet)
}

normalize <- function(x, maxv, minv) {
  return((x-minv)/(maxv-minv))
}


trainingdata <- read.csv(file="data/data_shuf_training.csv", header=F, sep=",")
trainingdata['V17'] = normalize(trainingdata['V17'], 0, 3)
inputSize <- 16
outputSize <- 1
seed <- 1

computeRank <- function(chromosome) {
  hiddenLayerSize <- bitsToInt(chromosome[4:(4+8)])
  denominator <- bitsToInt(chromosome[12:20])
  func <- chromosome[1:3]
  #set acctivaton functions
  actFunc = "logistic"
  if (func[1] == func[2]) {
    actFunc = "tanh"
  }
  outputLinear = T
  if (func[3] == 1) {
    outputLinear = F
  }

  #set weights
  set.seed(seed)
  weights <- rnorm((inputSize+1)*hiddenLayerSize+(hiddenLayerSize+1)*outputSize) / (denominator+1)

  #train perceptron
  net.sqrt <- neuralnet(
    V17~V1+V2+V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13+V14+V15+V16,
    trainingdata,
    threshold=0.1,
    stepmax=1e3,
    hidden=hiddenLayerSize,
    startweights = weights,
    linear.output = outputLinear,
    act.fct = actFunc
  )
  if (is.null(net.sqrt$weights)) {
    return(1e2*(hiddenLayerSize+1))
  }
  #compute result, the lower the better
  result <- net.sqrt$result.matrix['error',]*net.sqrt$result.matrix['steps',]
  return(result)
}

c(0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0)
