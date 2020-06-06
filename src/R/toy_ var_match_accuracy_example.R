library(dplyr)

df <- data.frame(
    c = rapply(list("x", "y", "z"), rep, times = 30),
    id_1 = unlist(lapply(c(1:9), rep, times = 10)),
    d2 = rapply(list(1:9), function(x) {sample(c(1:3), 10, replace = T)}),
    id_2   = unlist(lapply(c(1:9), function(x) {sample((1:9)[-x], 10, replace = T)})),
    cmt = sample(c(T,F), 90, replace = T),
    r1   = sample(1:10, 90, replace = T),
    r2   = sample(1:10, 90, replace = T),
    r3   = sample(1:10, 90, replace = T)
)



df <- df %>% group_by(c,id_1, d2) %>% 
    mutate(
        tpm = length(unique(id_1)),
        tpr_1 = ifelse(cmt == T & r1 == 1, 1, 0),
        tpr_5 = ifelse(cmt == T & r1 <= 5, 1, 0),
        tpr_10 = ifelse(cmt == T & r1 <= 10, 1, 0) 
    ) %>% group_by(c, id_1) %>%
    mutate(        
        pm_1 = length(r1 == 1),
        pm_5 = length(r1 <= 5),
        pm_10 = length(r1 <= 10)
        ) 


av <- df %>% group_by(c, id_1, id_2) %>% 
    mutate(
        tp_1_alt = sum(unique(tpr_1)),
        tp_5_alt  = sum(unique(tpr_5)),
        tp_10_alt  = sum(unique(tpr_10))
    ) %>% group_by(c, id_1) %>%
    summarise(
        tpm_alt = sum(length(id_1[cmt == T])*(length(unique(d2[cmt == T]))-1)),
        tpm = length(id_1[cmt == T]),
        tp_1 = sum(unique(tpr_1)),
        tp_5 = sum(unique(tpr_5)),
        tp_10 = sum(unique(tpr_10)),
        tp_1_alt = sum(unique(tpr_1)),
        tp_5_alt  = sum(unique(tpr_5)),
        tp_10_alt  = sum(unique(tpr_10)),
        pm_1 = sum(unique(pm_1)),
        pm_5 = sum(unique(pm_5)),
        pm_10 = sum(unique(pm_10))
    )


ac <- av  %>% group_by(c) %>% 
    summarise(tpm = sum(tpm),
              tpm_alt = sum(tpm_alt),
              r_1 =  sum(tp_1)/tpm,
              p_1 = sum(tp_1)/sum(pm_1),
              f1_1 =  (2*p_1*r_1)/(p_1+r_1),
              r_5 =  sum(tp_5)/tpm,
              p_5 = sum(tp_5)/sum(pm_5),
              f1_5 =  (2*p_5*r_5)/(p_5+r_5),
              r_10 =  sum(tp_10)/tpm,
              p_10 = sum(tp_10)/sum(pm_10),
              f1_10 =  (2*p_10*r_10)/(p_10+r_10),
              p_r_1 = sum(tp_1_alt)/tpm_alt, 
              p_r_5 = sum(tp_5_alt)/tpm_alt, 
              p_r_10 = sum(tp_10_alt)/tpm_alt, 
              f1_1 =  (2*p_r_1*p_r_1)/(p_r_1+p_r_1),
              f1_5 =  (2*p_r_5*p_r_5)/(p_r_5+p_r_5),
              f1_10 =  (2*p_r_10*p_r_10)/(p_r_10+p_r_10)
              )
              
              

