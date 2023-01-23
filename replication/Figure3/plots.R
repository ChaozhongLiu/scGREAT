# Go analysis on SOM discovered gene modules ----
#BiocManager::install('clusterProfiler')
library(clusterProfiler)
options(connectionObserver = NULL)
#BiocManager::install('org.Hs.eg.db')
library(org.Hs.eg.db)
library(ggplot2)
#BiocManager::install('DOSE')
library(DOSE)
rm(list=ls())

som_df <- read.csv('10X_CD4T.SOM_5n8.txt', stringsAsFactors=F,header=FALSE)
genes <- bitr(som_df$V1, fromType = 'SYMBOL', toType = c('ENSEMBL','ENTREZID'),  OrgDb=org.Hs.eg.db)

#GO
go <- enrichGO(gene = genes$ENSEMBL, OrgDb = "org.Hs.eg.db", ont="all", keyType = 'ENSEMBL')
go <- simplify(go)
dotplot(go, split="ONTOLOGY",showCategory=5,font.size=8) +facet_grid(~ONTOLOGY, scale="free")
write.table(go,"10X_CD4T.SOM5n8_GO.txt",quote=F,sep='\t')


# Trajectory curve ----
library(ggplot2)
library(ggpubr)
library(ggridges)
theme_set(theme_ridges())


color_palette_ct <- c('#433179',
                      '#458a8c',
                      '#fae855')


dat_1 <- read.csv('10X_CD4T.timecurve.reg.raw.csv')
colnames(dat_1) <- c('Cell_type', 'Pseudotime')
dat_1$data <- 'Regulatory State'
dat_2 <- read.csv('10X_CD4T.timecurve.rna.raw.csv')
colnames(dat_2) <- c('Cell_type', 'Pseudotime')
dat_2$data <- 'Transcriptome'

dat <- rbind(dat_1,dat_2)

ggplot(dat, aes(x=Pseudotime, fill=Cell_type)) +
  geom_density(aes(y=..count..),alpha=0.75) +
  #geom_area(stat = "bin", bins = 30, color = "black", alpha=0.75) +
  scale_fill_manual(name = "Cell type", 
                    values = color_palette_ct) +
  theme_bw() +
  xlab("Trajectory pseudotime")+
  ylab("Density")+
  facet_wrap(~data) +
  theme(text = element_text(size = 15))









