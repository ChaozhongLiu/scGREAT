library(fgsea)
library(ggplot2)

rm(list=ls())

# Gene list correlation from small to large ----
df <- read.csv(file = "results/10X.cd4T.GeneRank_L.txt",
               header=F, stringsAsFactors = F)
rnk_L <- df$V1 #create a list of correlation values from smallest to largest
names(rnk_L) <- df$V2 #gives each element a name (creates a named list)
rnk_L <- sort(rnk_L, decreasing = TRUE)

df <- read.csv(file = "results/10X.cd4T.GeneRank_r.txt",
               header=F, stringsAsFactors = F)
rnk_r <- df$V1 #create a list of correlation values from smallest to largest
names(rnk_r) <- df$V2 #gives each element a name (creates a named list)
rnk_r <- sort(rnk_r, decreasing = TRUE)



# Create the gmt file ----
gmt.file_c5 <- readLines("c5.all.v2022.1.Hs.symbols.gmt") #reads in each line of the file and creates a list
gmt.split_c5 <- strsplit(x = gmt.file_c5, split = "\t") #splits the line at each tab
#this creates a vector for each list element
gmt.names_c5 <- lapply(X = gmt.split_c5,
                    FUN = function(x){x[1]} ) #for each vector in the list take the first element
gmt.trimmed_c5 <- lapply(X=gmt.split_c5,
                      FUN=function(x){x[-1:-2]}) #for each vector in the list remove the first two elements
names(gmt.trimmed_c5)<-unlist(gmt.names_c5) #name each vector in the list with the pathway

gmt.file_c2 <- readLines("c2.cgp.v2022.1.Hs.symbols.gmt") #reads in each line of the file and creates a list
gmt.split_c2 <- strsplit(x = gmt.file_c2, split = "\t") #splits the line at each tab
#this creates a vector for each list element
gmt.names_c2 <- lapply(X = gmt.split_c2,
                       FUN = function(x){x[1]} ) #for each vector in the list take the first element
gmt.trimmed_c2 <- lapply(X=gmt.split_c2,
                         FUN=function(x){x[-1:-2]}) #for each vector in the list remove the first two elements
names(gmt.trimmed_c2)<-unlist(gmt.names_c2) #name each vector in the list with the pathway


# Run GSEA ----

run_gsea <- function(rnks){
  set.seed(42)
  fgseaRes_c2 <- fgsea(pathways = gmt.trimmed_c2,
                      stats = rnks,
                      minSize=15,
                      maxSize=500,
                      nperm=10000)
  set.seed(42)
  fgseaRes_c5 <- fgsea(pathways = gmt.trimmed_c5,
                       stats = rnks,
                       minSize=15,
                       maxSize=500,
                       nperm=10000)
  fgseaRes <- rbind(fgseaRes_c2,fgseaRes_c5)
  fgseaRes <- fgseaRes[order(fgseaRes$padj, -abs(fgseaRes$NES)),]
  fgseaRes$rank <- c(1:length(fgseaRes$padj))
  return(fgseaRes)
}

fgseaRes_L <- run_gsea(rnk_L)
fgseaRes_r <- run_gsea(rnk_r)


sum(fgseaRes_L[, padj < 0.01]) # 15/355
sum(fgseaRes_r[, padj < 0.01]) # 4/96


cc_pathways <- fgseaRes_L$pathway[grepl( 'CELL_CYCLE', fgseaRes_L$pathway,fixed = TRUE)]
tc_pathways <- fgseaRes_L$pathway[grepl( 'T_LYMPHOCYTE', fgseaRes_L$pathway,fixed = TRUE)]
tc_pathways <- c(tc_pathways, fgseaRes_L$pathway[grepl( '_T_CELL', fgseaRes_L$pathway,fixed = TRUE)])
  
cd4t_ranks_L <- fgseaRes_L[fgseaRes_L$pathway %in% tc_pathways, c('pathway','NES','rank')]
cd4t_ranks_r <- fgseaRes_r[fgseaRes_r$pathway %in% tc_pathways, c('pathway','NES','rank')]

rank_df_cd4t <- data.frame(ranks=c(cd4t_ranks_L$rank,cd4t_ranks_r$rank),
                      Method=c(rep('L',82), rep('r',82)))
# ECDF
ggplot(rank_df_cd4t, aes(ranks, color=Method)) + 
  stat_ecdf(geom = "step") + 
  xlim(0,9000) +
  xlab("GSEA pathway ranks") +
  ylab("ECDF") +
  theme_bw() +
  theme(text = element_text(size = 13)) # 6x4

# Bubble plot
pathways <- cd4t_ranks_L$pathway[1:15]
path_results_L <- fgseaRes_L[fgseaRes_L$pathway %in% pathways,]
path_results_L$Method <- 'L'
path_results_r <- fgseaRes_r[fgseaRes_r$pathway %in% pathways,]
path_results_r$Method <- 'r'

path_results <- rbind(path_results_L, path_results_r)
path_results$'-log10(padj)' <- -log10(path_results$padj)
path_results$'abs(NES)' <- abs(path_results$NES)
path_results$pathway <- factor(path_results$pathway, levels = rev(pathways))

ggplot(path_results, aes_string(x="Method", y="pathway", color=path_results$'-log10(padj)', size=path_results$'abs(NES)')) +
  geom_point() +
  scale_color_continuous(low="blue", high="red", name = '-log10(p.adj)', guide=guide_colorbar(reverse=TRUE)) +
  ylab(NULL) + ggtitle(NULL) + theme_bw() +
  scale_size_continuous(name='abs(NES)',range=c(1,5)) +
  #facet_wrap(~condition)+
  theme(axis.text.x = element_text(angle = 0, vjust = 0.5, hjust=1))

cc_ranks_L <- fgseaRes_L[fgseaRes_L$pathway %in% cc_pathways, 'rank']
cc_ranks_r <- fgseaRes_r[fgseaRes_r$pathway %in% cc_pathways, 'rank']

rank_df_cc <- data.frame(ranks=c(cc_ranks_L$rank,cc_ranks_r$rank),
                      method=c(rep('L',49), rep('r',49)))
ggplot(rank_df_cc, aes(ranks, color=method)) + 
  stat_ecdf(geom = "step") + 
  xlim(0,9000)


# Making some plots ----
plotEnrichment(pathway = gmt.trimmed[["ZHENG_FOXP3_TARGETS_IN_T_LYMPHOCYTE_DN"]],stats = rnk)





