#BiocManager::install('clusterProfiler')
library(clusterProfiler)
options(connectionObserver = NULL)
#BiocManager::install('org.Hs.eg.db')
library(org.Hs.eg.db)
library(ggplot2)
#BiocManager::install('DOSE')
library(DOSE)
rm(list=ls())



## Memory B v.s. Intermediate B ----
rm(list=ls())
degs_df <- read.csv('Spi1_genes.txt', stringsAsFactors=F,header=FALSE)
genes <- bitr(degs_df$V1, fromType = 'SYMBOL', toType = c('ENSEMBL','ENTREZID'),  OrgDb=org.Hs.eg.db)

#GO
go <- enrichGO(gene = genes$ENSEMBL, OrgDb = "org.Hs.eg.db", ont="BP", keyType = 'ENSEMBL')
go <- simplify(go)
dotplot(go, split="ONTOLOGY",showCategory=5,font.size=8) +facet_grid(~ONTOLOGY, scale="free")
barplot(go,showCategory=5) #8x6
write.table(go,"GO_Spi1.txt",quote=F,sep='\t')


## CD4 T markers ----
rm(list=ls())
degs_df <- read.csv('RUNX1_genes.txt', stringsAsFactors=F,header=FALSE)
genes <- bitr(degs_df$V1, fromType = 'SYMBOL', toType = c('ENSEMBL','ENTREZID'),  OrgDb=org.Hs.eg.db)

#GO
go <- enrichGO(gene = genes$ENSEMBL, OrgDb = "org.Hs.eg.db", ont="BP", keyType = 'ENSEMBL')
go <- simplify(go)
dotplot(go, split="ONTOLOGY",showCategory=5,font.size=8) #+facet_grid(~ONTOLOGY, scale="free")

barplot(go,showCategory=5) #8x6
write.table(go,"GO_RUNX1.txt",quote=F,sep='\t')





