cat("\014")
rm(list=ls())
#setwd("//HALNYDS/home/Dokumenty/Szkolenia/Programowanie w R")
setwd("C:/Users/Public/Documents/IT/Projects/Wiron")



d<-read.csv('dane.csv')

# test normalnosci
v<-(d$WIRON[-1]-d$WIRON[-nrow(d)])/d$WIRON[-nrow(d)]*100
v_additive<-(d$WIRON[-1]-d$WIRON[-nrow(d)])*100

wiron<-d$WIRON
polonia<-d$POLONIA
wiboron<-d$WIBOR_ON

hist(v, 50)
hist(log(v+100)-log(100), 50)

?chisq.test # test chi-kwadrat fischera

shapiro.test(v)
negative_viron<-d$WIRON<=0

# variance and 
var_wiron<-var(d$WIRON)
sd_wiron<-sd(d$WIRON)
mi_wiron<-mean(d$WIRON)
  
var_wiboron<-var(d$WIBOR_ON)
sd_wiboron<-sd(d$WIBOR_ON)
mi_wiboron<-mean(d$WIBOR_ON)

var_polonia<-var(d$POLONIA)
sd_polonia<-sd(d$POLONIA)
mi_polonia<-mean(d$POLONIA)

# CPD WIRON 1M and 3M are not calulated for all observation dates
# hence we need to compare WIBORS on same observation period

wiron_1m<-d[!is.na(d$CPD_1M_WIRON),'CPD_1M_WIRON']
var_wiron_1m<-var(wiron_1m)
sd_wiron_1m<-sd(wiron_1m)
mi_wiron_1m<-mean(wiron_1m)

var_wibor1m<-var(d$WIBOR_1M[1:length(wiron_1m)])
sd_wibor1m<-sd(d$WIBOR_1M[1:length(wiron_1m)])
mi_wibor1m<-mean(d$WIBOR_1M[1:length(wiron_1m)])


wiron_3m<-d[!is.na(d$CPD_3M_WIRON),'CPD_3M_WIRON']
var_wiron_3m<-var(wiron_3m)
sd_wiron_3m<-sd(wiron_3m)
mi_wiron_3m<-mean(wiron_3m)

var_wibor3m<-var(d$WIBOR_3M[1:length(wiron_3m)])
sd_wibor3m<-sd(d$WIBOR_3M[1:length(wiron_3m)])
mi_wibor3m<-mean(d$WIBOR_3M[1:length(wiron_3m)])

