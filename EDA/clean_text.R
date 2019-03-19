library(DBI)
library(RSQLite)
library(tidyverse)
library(gridExtra)
library(scales)
library(magrittr)
library(ggpirate)

#Get Data from Database
db.file <- sprintf("%s/%s/%s",
                   getwd(), 'Article Collection', 'articles_zenodo.db')
con <- dbConnect(SQLite(), dbname=db.file)
content <- dbGetQuery(con,'select * from train_content')
lean <- dbGetQuery(con,'select * from train_lean')
dbDisconnect(con)

#Duplicate articles ID
dup.file <- sprintf("%s/%s/%s", getwd(), 'Article Collection', "duplicate_article_ids_train.csv")
dup.articles <- read.csv(dup.file)

#Join data together
df <- left_join(content, lean, by='id') %>%
    mutate(Year = lubridate::year(`published-at`)) %>%
    filter(Year >= 2009, url_keep == 1,
           !(id %in% dup.articles$id.1)) %>%
    mutate(source = str_split(str_split(url, '//', simplify = T)[,2],
                              fixed('.'), simplify = T)[,1])

#View common contributors
df %>%
    group_by(source) %>%
    count(.) %>%
    View(.)

#Check foxbusiness text
df %>%
    filter(source == 'foxbusiness') %>%
    mutate(containURL = ifelse(grepl('fox business', text, ignore.case = T),
                               1, 0),
           containURL = ifelse(grepl('foxbusiness', text, ignore.case = T),
                               1, containURL)) %>%
    group_by(containURL) %>%
    count(.)


#Check source in text
df %>%
    rowwise() %>%
    mutate(containURL = ifelse(grepl(source, text, ignore.case = T),
                               1, 0)) %>%
    group_by(containURL) %>%
    count(.)

#Check AP in text
df %>%
    mutate(containURL = ifelse(grepl('(AP)', text, ignore.case = F),
                               1, 0)) %>%
    group_by(containURL) %>%
    count(.)
