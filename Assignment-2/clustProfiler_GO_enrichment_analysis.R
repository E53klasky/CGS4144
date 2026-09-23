setwd("/home/appu/Documents/school/uf_fall_2026/CGS4144/CGS4144/Assignment-2")

if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("clusterProfiler")
BiocManager::install("org.Hs.eg.db")

library(clusterProfiler)
library(org.Hs.eg.db)

# Load full DE results
de_results <- read.csv("results/differential_expression_all.csv")

# Filter to finalized significant gene list
# I am using 0.3 here because from DE.py when we chose 1 we only got 1 significant gene.
sig_genes <- de_results[de_results$adjusted_p_value < 0.05 & 
                          abs(de_results$log2_fold_change) >= 0.3, "gene"]

length(sig_genes)

# Convert Ensembl -> Entrez
gene_conversion <- bitr(
  sig_genes,
  fromType = "ENSEMBL",
  toType = "ENTREZID",
  OrgDb = org.Hs.eg.db
)

nrow(gene_conversion)  # how many survived conversion

failed_genes <- setdiff(sig_genes, gene_conversion$ENSEMBL)

length(failed_genes)
length(failed_genes) / length(sig_genes)

head(failed_genes)

all_genes <- de_results$gene

global_conversion <- bitr(
  all_genes,
  fromType = "ENSEMBL",
  toType = "ENTREZID",
  OrgDb = org.Hs.eg.db
)

global_entrez <- global_conversion$ENTREZID

length(all_genes)
length(global_entrez)

go_results <- enrichGO(
  gene = gene_conversion$ENTREZID,
  universe = global_entrez,
  OrgDb = org.Hs.eg.db,
  ont = "BP",
  pAdjustMethod = "BH",
  pvalueCutoff = 0.05,
  qvalueCutoff = 0.05,
  readable = TRUE
)

head(as.data.frame(go_results))
