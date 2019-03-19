library(DBI)
library(RSQLite)
library(tidyverse)
library(gridExtra)
library(scales)
library(magrittr)

#Get Data from Database
db.file <- sprintf("%s/%s/%s", getwd(), 'Article Collection', "articles_zenodo.db")
con <- dbConnect(SQLite(), dbname=db.file)
lean <- dbGetQuery(con,'select * from train_lean')
content <- dbGetQuery(con,'select * from train_content')

#Duplicate articles ID
dup.file <- sprintf("%s/%s/%s", getwd(), 'Article Collection', "duplicate_article_ids_train.csv")
dup.articles <- read.csv(dup.file)

#Get just the stuff we want
reduced.df <-
    left_join(content, lean, by='id') %>%
    mutate(Year = lubridate::year(`published-at`),
           Year = ifelse(is.na(Year), 2021, Year)) %>%
    filter(!is.na(`published-at`),
           Year >= 2009,
           url_keep == 1,
           !(id %in% dup.articles$id.1))

#Get publishers with at least 100 articles
pubs.g100 <-
    reduced.df %>%
    mutate(source = str_split(str_split(url, '//', simplify = T)[,2],
                              fixed('.'), simplify = T)[,1],
           bias_final = as.character(bias_final),
           bias_final = ifelse(bias_final == 'left-center', 'least', bias_final),
           bias_final = ifelse(bias_final == 'right-center', 'least', bias_final),
           bias_final = factor(bias_final, levels = c('left', 'least', 'right'))) %>%
    filter(!(source %in% c('foxbusiness', 'apnews'))) %>%
    group_by(bias_final, source) %>%
    count() %>%
    filter(n >= 100) %>%
    mutate(pubs_100 = 1) %>%
    ungroup() %>%
    select(source, pubs_100)

#Update lean table
lean.update <-
    lean %>%
    mutate(source = str_split(str_split(url, '//', simplify = T)[,2],
                              fixed('.'), simplify = T)[,1]) %>%
    full_join(., pubs.g100, by='source')

dbWriteTable(conn = con, name = 'train_lean', value = lean.update,
             row.names=F, overwrite=T)

dbDisconnect(con)
