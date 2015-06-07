if (!require("genalg")) {
  install.packages("genalg", repos="http://cran.rstudio.com/")
  install.packages("ggplot2", repos="http://cran.rstudio.com/")
  library(genalg)
  library(ggplot2)
}
source("script/perceptron.r")

# http://stackoverflow.com/a/25411493/1387612
bitsToInt<-function(x) {
  packBits(c(as.logical(x), rep(FALSE, 32-length(x)%%32)), "integer")
}

# Example chromosome from documentation
chromosome = c(
  # funkcja aktywacji – 3 elementowy wektor binarny okręslający funkcje wyjścia poszczególnych warstw
  0,1,0,
  # ilość neuronów w warstwie ukrytej – dodatnia liczba całkowita (5)
  1,0,1,0,0,0,0,0,
  # stała przez którą dzielimy wagi
  1,1,0,0,0,0,0,0,0
)

evalFunc <- function(x) {
    print(x)
    hiddenLayerSize <- bitsToInt(x[4:(4+8)])
    denominator <- bitsToInt(x[12:20])
    return(computeRank(hiddenLayerSize, x[1:3], denominator))
}

iter = 1
GAmodel <- rbga.bin(size = 20, popSize = 5, iters = iter, mutationChance = 0.0001,
    elitism = T, evalFunc = evalFunc)
cat(summary(GAmodel))
warnings()
png(filename="plot.png")
plot(GAmodel)
dev.off()
png(filename="hist.png")
plot(GAmodel, type='hist')
dev.off()
