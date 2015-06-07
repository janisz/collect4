if (!require("neuralnet")) {
  install.packages('neuralnet', repos="http://cran.rstudio.com/")
  library(neuralnet)
}
if (!require("datasets")) {
  install.packages('datasets', repos="http://cran.rstudio.com/")
  library(datasets)
}

normalize <- function(x, maxv, minv) {
  return((x-minv)/(maxv-minv))
}



computeRank <- function(hiddenLayerSize, bias, func, seed) {

  traininginput <-  as.data.frame(runif(50, min=0, max=100))
  trainingoutput <- sqrt(traininginput)
  trainingdata <- cbind(traininginput,normalize(trainingoutput, 0, 10))
  colnames(trainingdata) <- c("Input","Output")
  testdata <- as.data.frame((1:10)^2)

  inputSize <- 1
  hiddenLayerSize <- 10
  outputSize <- 1
  bias <- c(0,1,1)
  func <- c(0,0,1)
  seed <- 1

  #exclude bias weights
  exclude <- NULL
  if(bias[2] == 1) {
    layer <- seq(from = 1, to = 1, length.out = hiddenLayerSize)
    from <- seq(from = 1, to = 1, length.out = hiddenLayerSize)
    to  <- seq(from = 1, to = hiddenLayerSize, length.out = hiddenLayerSize)
    exclude <- rbind(matrix(c(layer, from, to), nrow=length(layer)), exclude)
  }
  if(bias[3] == 1) {
    layer <- seq(from = 2, to = 2, length.out = outputSize)
    from <- seq(from = 1, to = 1, length.out = outputSize)
    to  <- seq(from = 1, to = outputSize, length.out = outputSize)
    exclude <- rbind(matrix(c(layer, from, to), nrow=length(layer)), exclude)
  }

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
  weights <- rnorm((inputSize+1)*hiddenLayerSize+hiddenLayerSize*outputSize+1)

  #train perceptron
  net.sqrt <- neuralnet(
    Output~Input,
    trainingdata,
    threshold=0.1,
    hidden=hiddenLayerSize,
    exclude = exclude,
    startweights = weights,
    linear.output = outputLinear,
    act.fct = actFunc
  )
  #compute result the lower the better
  return(net.sqrt$result.matrix['error',]*net.sqrt$result.matrix['steps',])
}
