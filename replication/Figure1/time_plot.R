setwd('~/../data/thesis/')
library(ggplot2)
library(ggpubr)

color_palette_m <- c("#7cef61", #vec
                     "#d72628") #esda

time_GL <- read.csv('time.globalL.csv')
time_GL <- time_GL[time_GL$n_bulks==5000,]
time_GL$index <- 'Global L'

time_LL <- read.csv('time.localL.csv')
time_LL <- time_LL[time_LL$n_bulks==5000,]
time_LL$index <- 'Local L'

time_MI <- read.csv('time.moransi.csv')
time_MI <- time_MI[time_MI$n_bulks==5000,]
time_MI$index <- "Moran's I"

time_df <- rbind(time_GL,time_LL,time_MI)
time_df$index <- factor(time_df$index,
                        levels = c("Moran's I",'Global L','Local L'))
time_df$method <- factor(time_df$method, levels=c('sc','esda'))

ggplot(time_df, aes(x=n_features, y=time,color=method)) +
  #geom_line(size=1) +
  geom_point(size=0.5,color='black')+
  #geom_ribbon(alpha=0.5)+
  scale_color_manual(name = "method", 
                     values = color_palette_m) +
  theme_bw() +
  xlab("#Features")+#scale_x_log10()+
  ylab("Time")+
  scale_y_log10(breaks=c(0,1,10,60,600,1800,7200),
                labels=c('0s','1s','10s','1min','10min','30min','2h'))+
  scale_x_continuous(breaks=c(0,1000,2000,5000,10000),
                labels=c('0','1k','2k','5k','10k'))+
  stat_smooth()+
  facet_wrap(.~index, ncol = 1) +
  theme(text = element_text(size = 18))
#5x9

