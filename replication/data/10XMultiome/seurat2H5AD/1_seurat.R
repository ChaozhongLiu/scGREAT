library(SeuratData)
library(Seurat)
library(dplyr)
library(Signac)
library(EnsDb.Hsapiens.v86)
library(ggplot2)
library(cowplot)
library(hdf5r)
#library(rhdf5)
rm(list=ls())

#integrated seurat object to get QC and cell type annotation ----
# the 10x hdf5 file contains both data types. 
inputdata.10x <- Read10X_h5("../raw/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5")

# extract RNA and ATAC data
rna_counts <- inputdata.10x$`Gene Expression`
atac_counts <- inputdata.10x$Peaks

# Create Seurat object
pbmc <- CreateSeuratObject(counts = rna_counts)
pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "^MT-")

# Now add in the ATAC-seq data
# we'll only use peaks in standard chromosomes
grange.counts <- StringToGRanges(rownames(atac_counts), sep = c(":", "-"))
grange.use <- seqnames(grange.counts) %in% standardChromosomes(grange.counts)
atac_counts <- atac_counts[as.vector(grange.use), ]
annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)
seqlevelsStyle(annotations) <- 'UCSC'
genome(annotations) <- "hg38"

frag.file <- "../raw/pbmc_granulocyte_sorted_10k_atac_fragments.tsv.gz"
chrom_assay <- CreateChromatinAssay(
  counts = atac_counts,
  sep = c(":", "-"),
  genome = 'hg38',
  fragments = frag.file,
  min.cells = 10,
  annotation = annotations
)
pbmc[["ATAC"]] <- chrom_assay


VlnPlot(pbmc, features = c("nCount_ATAC", "nCount_RNA","percent.mt"), ncol = 3,
        log = TRUE, pt.size = 0) + NoLegend()
pbmc <- subset(
  x = pbmc,
  subset = nCount_ATAC < 7e4 &
    nCount_ATAC > 5e3 &
    nCount_RNA < 25000 &
    nCount_RNA > 1000 &
    percent.mt < 20
)

cells_passed_QC <- colnames(pbmc)
#rm(pbmc)
DefaultAssay(pbmc) <- "RNA"
pbmc <- SCTransform(pbmc, verbose = FALSE)
pbmc <- RunPCA(pbmc)
pbmc <- RunUMAP(pbmc,dims = 1:50, reduction.name = 'umap.rna', reduction.key = 'rnaUMAP_')

# ATAC analysis
# We exclude the first dimension as this is typically correlated with sequencing depth
DefaultAssay(pbmc) <- "ATAC"
pbmc <- RunTFIDF(pbmc)
pbmc <- FindTopFeatures(pbmc, min.cutoff = 'q0')
pbmc <- RunSVD(pbmc)
pbmc <- RunUMAP(pbmc, reduction = 'lsi', dims = 2:50, reduction.name = "umap.atac", reduction.key = "atacUMAP_")

pbmc <- FindMultiModalNeighbors(pbmc, reduction.list = list("pca", "lsi"), dims.list = list(1:50, 2:50),k.nn = 20)
pbmc <- RunUMAP(pbmc, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_")
pbmc <- FindClusters(pbmc, graph.name = "wsnn", algorithm = 3, verbose = FALSE)
pbmc <- FindSubCluster(pbmc, cluster = 6, graph.name = "wsnn", algorithm = 3)
Idents(pbmc) <- "sub.cluster"
pbmc <- RenameIdents(pbmc, '19' = 'pDC','20' = 'HSPC','15' = 'cDC')
pbmc <- RenameIdents(pbmc, '0' = 'CD14 Mono', '9' ='CD14 Mono', '5' = 'CD16 Mono')
pbmc <- RenameIdents(pbmc, '17' = 'Naive B', '11' = 'Intermediate B', '10' = 'Memory B', '21' = 'Plasma')
pbmc <- RenameIdents(pbmc, '7' = 'NK')
pbmc <- RenameIdents(pbmc, '4' = 'CD4 TCM', '13'= "CD4 TEM", '3' = "CD4 TCM", '16' ="Treg", '1' ="CD4 Naive", '14' = "CD4 Naive")
pbmc <- RenameIdents(pbmc, '2' = 'CD8 Naive', '8'= "CD8 Naive", '12' = 'CD8 TEM_1', '6_0' = 'CD8 TEM_2', '6_1' ='CD8 TEM_2', '6_4' ='CD8 TEM_2')
pbmc <- RenameIdents(pbmc, '18' = 'MAIT')
pbmc <- RenameIdents(pbmc, '6_2' ='gdT', '6_3' = 'gdT')
pbmc$celltype <- Idents(pbmc)

seurat_annotation <- pbmc$celltype
all(names(seurat_annotation) == cells_passed_QC)

DimPlot(pbmc, group.by = "celltype", label = TRUE) #+ NoLegend()

pbmc <- FindMultiModalNeighbors(pbmc, reduction.list = list("pca", "lsi"),
                                dims.list = list(1:50, 2:50),k.nn = 49)
idx_mtx <- pbmc@neighbors$weighted.nn@nn.idx
dist_mtx <- pbmc@neighbors$weighted.nn@nn.dist
seurat_wnn_graph <- cbind(idx_mtx,dist_mtx)
rownames(seurat_wnn_graph) <- pbmc@neighbors$weighted.nn@cell.names
write.table(seurat_wnn_graph,'../seurat_wnn_graph.csv',quote=F,
            col.names = F, sep=',')
saveRDS(pbmc, 'pbmc.wnn.rds')

pbmc <- readRDS('pbmc.wnn.rds')


#create splited seurat object ----
pbmc.rna <- CreateSeuratObject(counts = rna_counts)
pbmc.rna[["percent.mt"]] <- PercentageFeatureSet(pbmc.rna, pattern = "^MT-")
pbmc.atac <- CreateSeuratObject(counts = chrom_assay, assay = "ATAC")

#apply the QC done before ----
pbmc.rna@meta.data$seurat_annotations <- 'none'
pbmc.rna@meta.data[!colnames(pbmc.rna) %in% cells_passed_QC,]$seurat_annotations <- 'filtered'
summary(factor(pbmc.rna$seurat_annotations))
pbmc.rna <- subset(pbmc.rna, seurat_annotations != "filtered")

pbmc.atac@meta.data$seurat_annotations <- 'none'
pbmc.atac@meta.data[!colnames(pbmc.atac) %in% cells_passed_QC,]$seurat_annotations <- 'filtered'
summary(factor(pbmc.atac$seurat_annotations))
pbmc.atac <- subset(pbmc.atac, seurat_annotations != "filtered")
all(colnames(pbmc.rna) == colnames(pbmc.atac))

all(names(seurat_annotation) == colnames(pbmc.atac))
all(names(seurat_annotation) == colnames(pbmc.rna))

pbmc.rna$seurat_annotations <- seurat_annotation
pbmc.atac$seurat_annotations <- seurat_annotation
summary(pbmc.rna$seurat_annotations)


# Perform standard analysis of each modality independently RNA analysis ----
pbmc.rna <- NormalizeData(pbmc.rna)
pbmc.rna <- FindVariableFeatures(pbmc.rna)
pbmc.rna <- ScaleData(pbmc.rna)
pbmc.rna <- RunPCA(pbmc.rna)
pbmc.rna <- RunUMAP(pbmc.rna, dims = 1:30)
pbmc.rna <- FindNeighbors(pbmc.rna, dims = 1:30)
pbmc.rna <- FindClusters(pbmc.rna, resolution = 0.9, graph.name = 'RNA_snn',algorithm=3)
DimPlot(pbmc.rna, label=TRUE)

# ATAC analysis add gene annotation information
annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)
seqlevelsStyle(annotations) <- "UCSC"
genome(annotations) <- "hg38"
Annotation(pbmc.atac) <- annotations

# We exclude the first dimension as this is typically correlated with sequencing depth
pbmc.atac <- RunTFIDF(pbmc.atac)
pbmc.atac <- FindTopFeatures(pbmc.atac, min.cutoff = "q0")
pbmc.atac <- RunSVD(pbmc.atac)
pbmc.atac <- RunUMAP(pbmc.atac, reduction = "lsi", dims = 2:30, reduction.name = "umap.atac", reduction.key = "atacUMAP_")

pbmc.atac <- FindNeighbors(pbmc.atac, dims = 2:30, reduction = 'lsi')
pbmc.atac <- FindClusters(pbmc.atac, resolution = 0.8,graph.name = 'ATAC_snn',algorithm=3)
summary(pbmc.atac@meta.data$seurat_clusters) #4

#UMAP plot
p1 <- DimPlot(pbmc.rna, group.by = "seurat_annotations", label = TRUE) + NoLegend() + ggtitle("RNA")
DimPlot(pbmc.atac, group.by = "orig.ident", label = FALSE) + NoLegend() + ggtitle("ATAC")
p1 <- DimPlot(pbmc.atac, label = TRUE)
p2 <- DimPlot(pbmc.atac,group.by = "seurat_annotations", label = TRUE)
p1 #| p2


# quantify gene activity
gene.activities <- GeneActivity(pbmc.atac, features = VariableFeatures(pbmc.rna))

# add gene activities as a new assay
pbmc.atac[["ACTIVITY"]] <- CreateAssayObject(counts = gene.activities)

# normalize gene activities
DefaultAssay(pbmc.atac) <- "ACTIVITY"
pbmc.atac <- NormalizeData(pbmc.atac)
pbmc.atac <- ScaleData(pbmc.atac, features = rownames(pbmc.atac))

saveRDS(pbmc.rna, 'pbmc.rna.rds')
saveRDS(pbmc.atac, 'pbmc.atac.rds')

# save umap embedding
rna_umap <- rna@reductions$umap@cell.embeddings
write.table(rna_umap, '../PBMC.rna.umap.csv', sep=',',quote=F)
atac_umap <- atac@reductions$umap@cell.embeddings
write.table(atac_umap, '../PBMC.atac.umap.csv', sep=',',quote=F)




# save h5 file for AnnData object creation ----
library(Seurat)
library(ggplot2)
library(patchwork)
library(hdf5r)
library(rhdf5)
library(pheatmap)
library(RColorBrewer)
#library(reticulate)
set.seed(14)
#use_python('/usr/lib/python3.6')
#library(anndata)

rna.pbmc <- readRDS('pbmc.rna.rds')
rna_mtx <- Matrix::t(rna.pbmc[['RNA']]@counts)
rna_mtx <- as.matrix(rna_mtx)
h5createFile("rna.pbmc.h5")
h5write(rna_mtx, 'rna.pbmc.h5', 'RNA')
rna_meta <- rna.pbmc@meta.data
write.table(rna_meta, 'rna.meta.csv',quote = F, sep='\t')
rna_gene <- rownames(rna.pbmc)
write.table(rna_gene, 'rna_gene_name.txt',quote=F,row.names=F,col.names=F)

atac.pbmc <- readRDS('pbmc.atac.rds')
#all(colnames(rna.pbmc) == colnames(atac.brain))
#TRUE
gene.activities <- GeneActivity(atac.pbmc, features = rownames(rna.pbmc))
# add gene activities as a new assay
atac.pbmc[["ACTIVITY"]] <- CreateAssayObject(counts = gene.activities)

atac_mtx <- Matrix::t(atac.pbmc[['ACTIVITY']]@counts)
atac_mtx <- as.matrix(atac_mtx)
h5createFile("atac.pbmc.h5")
h5write(atac_mtx, 'atac.pbmc.h5', 'ATAC')
atac_meta <- atac.pbmc@meta.data
write.table(atac_meta, 'atac.meta.csv',quote = F, sep='\t')
DefaultAssay(atac.pbmc) <- 'ACTIVITY'
#all(rownames(atac.pbmc)==rownames(rna.pbmc))
#FLASE
atac_gene <- rownames(atac.pbmc)
write.table(atac_gene, 'atac_gene_name.txt',quote=F,row.names=F,col.names=F)

#peaks matrix
atac_mtx <- Matrix::t(atac.pbmc@assays$ATAC@counts)
atac_mtx <- as.matrix(atac_mtx)
h5createFile("atac.pbmc.peak.h5")
h5write(atac_mtx[1:2000,], 'atac.pbmc.peak.h5', 'ATAC-1')
h5write(atac_mtx[2001:4000,], 'atac.pbmc.peak.h5', 'ATAC-2')
h5write(atac_mtx[4001:6000,], 'atac.pbmc.peak.h5', 'ATAC-3')
h5write(atac_mtx[6001:8000,], 'atac.pbmc.peak.h5', 'ATAC-4')
h5write(atac_mtx[8001:10000,], 'atac.pbmc.peak.h5', 'ATAC-5')
h5write(atac_mtx[10001:10412,], 'atac.pbmc.peak.h5', 'ATAC-6')

DefaultAssay(atac.pbmc) <- 'ATAC'
atac_peak <- rownames(atac.pbmc)
write.table(atac_peak, 'atac_peak_name.txt',quote=F,row.names=F,col.names=F)








