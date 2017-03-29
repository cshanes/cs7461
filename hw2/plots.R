data = read.csv("/Users/chanes/gatech/cs7461/ABAGAIL/four_peaks/four_peaks.csv",header=TRUE,sep=",",stringsAsFactors=FALSE)

library(ggplot2)

#1.a
ggplot(data=data, aes(x=t, y=optimum, color=algorithm)) +
  geom_line() + 
  xlab("T") + 
  ylab("Optimum value (avg)") + 
  ggtitle("Algorithm performance")


knapsack_data = read.csv("/Users/chanes/gatech/cs7461/ABAGAIL/four_peaks/knapsack/Untitled/Knapsack.csv",header=TRUE,sep=",",stringsAsFactors=FALSE)
#1.a
ggplot(data=knapsack_data, aes(x=items, y=optimum, color=algorithm)) +
  geom_line() + 
  xlab("Num items") + 
  ylab("Optimum value (avg)") + 
  ggtitle("Algorithm performance")

nn_data = read.csv("/Users/chanes/gatech/cs7461/ABAGAIL/nn/nn_errors.csv",header=TRUE,sep=",",stringsAsFactors=FALSE)
ggplot(data=nn_data, aes(x=Iteration, y=Error, color=Algorithm)) +
  geom_line() + 
  xlab("Iteration") + 
  ylab("Error") + 
  ggtitle("Error results")

co_data = read.csv("/Users/chanes/gatech/cs7461/ABAGAIL/countones.csv",header=TRUE,sep=",",stringsAsFactors=FALSE)
ggplot(data=co_data, aes(x=n, y=diff, color=algorithm)) +
  geom_line() + 
  xlab("N") + 
  ylab("Diff from N (avg)") + 
  ggtitle("Algorithm performance")

ggplot(data=co_data, aes(x=n, y=log(time), color=algorithm)) +
  geom_line() + 
  xlab("N") + 
  ylab("log(time)") + 
  ggtitle("Algorithm time")
