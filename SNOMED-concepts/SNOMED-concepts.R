library(Rdiagnosislist)

SNOMED <- loadSNOMED("SNOMED/SnomedCT_USEditionRF2_PRODUCTION_20210901T120000Z/Full/Terminology/")

hf <- SNOMEDconcept('Heart failure', SNOMED = SNOMED)

description(hf, include_synonyms = TRUE, SNOMED = SNOMED)

hf_bad <- SNOMEDconcept('123', SNOMED = sampleSNOMED())
hf_bad_dsc <- description(hf_bad, include_synonyms = TRUE, SNOMED = sampleSNOMED())


getDescription <- function(x) {
  hf <- SNOMEDconcept(x, SNOMED = SNOMED)
  description(hf, include_synonyms = FALSE, SNOMED = SNOMED)
}
valid_dsc <- function(x) {
  length(x$term) > 0
}

valid_dsc(hf_dsc)
length(x[1]$term)
hf_dsc$term

terms <- data$concept_name
terms <- c("Heart failure", "Heart failure with reduced ejection fraction")
terms
x <- lapply(terms, getDescription)
filt_x <- Filter(valid_dsc, x)
View(terms)
rbind_x <- rbind(x)

library(dplyr)
data$concept_name[234147:234167]
data <- as_tibble(read.delim("OHDSI/CONCEPT.csv", sep="\t"))
concept_classes <- c( "Observable Entity", 
                      "Clinical Drug Form", 
                      "Clinical Finding",
                      "Procedure",
                      "Substance")
as_tibble(data) %>% filter(concept_class_id %in% concept_classes) %>%
  filter(invalid_reason == "")

final_data <- as_tibble(data) %>% filter(concept_class_id %in% concept_classes) %>%
  filter(valid_end_date == "20991231")

write.csv()

library(bit64)
descriptions <- description(as.integer64(data$concept_id[1]), include_synonyms = FALSE, SNOMED = SNOMED)

SNOMEDconcept(as.integer64(84114007), SNOMED = SNOMED)
as.integer64(data$concept_id)
domain <- read.delim("SNOMED/DOMAIN.csv", sep="\t")
typeof(84114007)

library(dplyr)

data_domain <- left_join(domain, data, by = "domain_id")

write.csv(final_data, sep = ",", file="output/CONCEPT_with_classes.csv")
