library(DBI)
library(RSQLite)
library(tidyverse)

db.file <- sprintf("%s/%s",
                   str_replace(dirname(rstudioapi::getSourceEditorContext()$path),
                               'EDA', 'Article Collection'),
                   "articles_zenodo.db")

con <- dbConnect(SQLite(), dbname=db.file)
lean <- dbGetQuery(con,'select * from lean')

#Plot th bias
lean %>%
    mutate(bias = factor(bias,
                         levels = c('left', 'left-center',
                                    'least',
                                    'right-center', 'right'))) %>%
    ggplot() +
    geom_bar(aes(x=bias, fill=bias), stat = 'count') +
    geom_text(stat='count',
              aes(x=bias, label=..count..),
              vjust=1,
              color=c('white', 'grey40', 'grey40', 'grey40', 'white')) +
    scale_fill_manual(values = c('dodgerblue2', 'lightblue',
                                 'gray',
                                 'indianred1', 'firebrick')) +
    theme_grey() +
    theme(legend.position = 'none') +
    labs(x='Article Bias', y='# of Articles')

