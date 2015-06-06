if (!require("genalg")) {
  install.packages("genalg", repos="http://cran.rstudio.com/")
  install.packages("ggplot2", repos="http://cran.rstudio.com/")
  library(genalg)
  library(ggplot2)
}

# http://stackoverflow.com/a/25411493/1387612
bitsToInt<-function(x) {
  packBits(c(as.logical(x), rep(FALSE, 32-length(x)%%32)), "integer")
}

# Example chromosome from documentation
chromosome = c(
  # funkcja aktywacji – 3 elementowy wektor binarny okręslający funkcje wyjścia poszczególnych warstw
  0,1,0,
  # ilość neuronów w warstwie ukrytej – dodatnia liczba całkowita (5)
  1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  # wektor wag – dodatnia liczba całkowita określająca ziarno losowania wektora wag (2356)
  0,0,1,0,1,1,0,0,1,0,0,1,0,0,0,0,0,
  # bias – 3 elementowy wektor binarny określający obecność biasu dla poszczególnych warstw
  0,1,1
)

evalFunc <- function(x) {
    x[1]
    x[2]
    x[3]

    hiddenLayerSize <- bitsToInt(x[4:(4+16)])
    seed <- bitsToInt(x[(4+17):(4+17+16)])

    x[4+17+17]
    x[4+17+17+1]
    x[4+17+17+2]

    #TODO: Here we should build and train perceptron. We should return error on test set

    return(hiddenLayerSize-seed)
}

iter = 100
GAmodel <- rbga.bin(size = 40, popSize = 200, iters = iter, mutationChance = 0.0001,
    elitism = T, evalFunc = evalFunc)
cat(summary(GAmodel))

png(filename="plot.png")
plot(GAmodel)
dev.off()
png(filename="hist.png")
plot(GAmodel, type='hist')
dev.off()
