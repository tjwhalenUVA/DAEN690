library(DBI)
library(RSQLite)
library(tidyverse)
library(gridExtra)
library(tidytext)
#Get Data
db.file <- sprintf("%s/Article Collection/%s", getwd(), "articles_zenodo.db")
con <- dbConnect(SQLite(), dbname=db.file)
lean <- dbGetQuery(con,'select * from train_lean')
content <- dbGetQuery(con,'select * from train_content limit 30000')
dbDisconnect(con)
#Join Content and leaning
df <- left_join(content, lean, by='id') %>%
    filter(url_keep == 1,
           lubridate::year(`published-at`) >= 2009) %>%
    select(id, title, bias_final)

#Tokenize Words in Title
title_words <- df %>%
    unnest_tokens(word, title) %>%
    group_by(id, bias_final, word) %>%
    count(sort = TRUE) %>%
    ungroup()
#Calculate TF-IDF
title_idf <- title_words %>% bind_tf_idf(word, id, n)
#Small Sample for Calculating Cor
sdf <-
    title_idf %>%
    select(id, word, tf_idf) %>%
    filter(id %in% unique(title_idf$id)[1:1000]) %>%
    spread(id, tf_idf, fill=0) %>%
    select(-word)

sim.df <- as.data.frame(cor(sdf)) %>% 
    rownames_to_column("id1") %>%
    gather(id2, correlation, -id1) %>%
    filter(id1 != id2, 
           correlation != 1) %>%
    arrange(desc(correlation))

head(sim.df)

content %>%
    select(id, title, text) %>%
    filter(id %in% c(60606, 43743)) %>%
    View(.)
