#%%
library(caret)
library(randomForest)
library(compositions) 
library(ggplot2)
library(readr)
library(tibble)

#1.Data filtering

#Read abundance file
abundance <- read_tsv("/Users/lawrenceadu-gyamfi/Documents/PERSONAL/PROJECTS/ML_Microbiome_Package/Dataset/abundance_crc.txt")
otu_table <- column_to_rownames(abundance, var = "Species")

#Read metadata file
metadata <- read_tsv("/Users/lawrenceadu-gyamfi/Documents/PERSONAL/PROJECTS/ML_Microbiome_Package/Dataset/metadata_crc.txt")
sample.labels <- as.factor(na.omit(metadata$Group))

#%%
#Example of thresholds for abundance and prevalence
abundance_cutoff <- 0.000001
prevalence_cutoff <- 0.05

# otu_table <- abundance

mean_abundance <- rowMeans(otu_table)


taxa_abundance_filtered <- names(mean_abundance[mean_abundance >= abundance_cutoff])
otu_table_abundance_filtered <- otu_table[taxa_abundance_filtered, , drop = FALSE]


prevalence <- rowMeans(otu_table_abundance_filtered > 0)  

#Data filtering using thresholds to obtain final dataframe for downstream data normalization
taxa_prevalence_filtered <- names(prevalence[prevalence >= prevalence_cutoff])
otu_table_final <- otu_table_abundance_filtered[taxa_prevalence_filtered, , drop = FALSE]
#%%

#2.Data normalization
#centered-log_ratio_transformation
clr_transform <- function(data, log.n0 = 1e-06) {
  data[data == 0] <- log.n0  # Replace all zeroes to avoid log(0) which is undefined
  clr_data <- clr(data)      
  return(clr_data)
}


otu_table_clr <- clr_transform(otu_table_final)

#3.Model training

#Setting cross_validation parameters
set.seed(42)
train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 2)

#Training the model using Random Forest
# x <- as.matrix(otu_table_clr) 
x <- t(as.matrix(otu_table_clr))
y <- sample.labels             #Needs to be a factors (metadata) that correspond to the samples
#%%
rf_model <- train(x, y,
                  method = "rf",
                  trControl = train_control,
                  tuneLength = 5,
                  ntree=100)

#%%


pred_probs <- predict(rf_model, x, type = "prob")


#4.Model evaluation

#Plotting ROC curves
