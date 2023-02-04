loadNBP<-function(year = "2021"){
  nazwa<-paste("https://www.nbp.pl/kursy/Archiwum/archiwum_tab_a_", year, ".csv", sep="")
  d<-read.csv(nazwa, sep=";")
  d<-d[,1:36]
  for(i in 1:ncol(d)){
    d[,i]<-gsub(",",".",d[,i])
  }
  for(i in 2:ncol(d)){
    d[,i]<-as.numeric(d[,i])
  }
  for(i in 2:ncol(d)){
    d[,i]<-d[,i]/d[nrow(d),i]
  }
  d$data<-as.Date(d$data, "%Y%m%d")
  for(i in 1:ncol(d)){
    d<-d[!is.na(d[,i]),]
  }
  return(d)
}