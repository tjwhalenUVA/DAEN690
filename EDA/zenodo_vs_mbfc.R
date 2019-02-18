#Load packages
library(DBI)
library(RSQLite)
library(tidyverse)

#set up wkdir path
wkng.dir <- dirname(rstudioapi::getSourceEditorContext()$path)
bias.levels <- c('left', 'leftcenter', 'least', 'rightcenter', 'right')

#Connect to zenodo database
db.file.z <- sprintf("%s/%s", str_replace(wkng.dir, 'EDA', 'Article Collection'), "articles_zenodo.db")
con.z <- dbConnect(SQLite(), dbname=db.file.z)
z <- dbGetQuery(con.z,'select * from lean')
dbDisconnect(con.z)
#Connect to Media Bias Fact Checker database
db.file.m <- sprintf("%s/%s", str_replace(wkng.dir, 'EDA', 'MBFC Scraper'), "mbfc.db")
con.m <- dbConnect(SQLite(), dbname=db.file.m)
m <- dbGetQuery(con.m,'select * from bias')
dbDisconnect(con.m)

#Compare the lean between the two
z %>%
    mutate(source = str_split(str_split(url, '//', simplify = T)[,2], fixed('.'), simplify = T)[,1],
           bias = str_replace(bias, '-', '')) %>%
    select(id, source, zenodo = bias) %>%
    left_join(.,
              m %>%
                  mutate(source = str_replace(sourceURL, fixed('www.'), ''),
                         source = str_split(source, fixed('//'), simplify = T)[,2],
                         source = str_split(source, fixed('.'), simplify = T)[,1]) %>%
                  select(source, mbfc = bias),
              by='source')%>%
    mutate(zenodo = factor(zenodo, levels = bias.levels),
           mbfc = factor(mbfc, levels = bias.levels))



