library(DBI)
library(RSQLite)
library(tidyverse)

db.file <- sprintf("%s/%s",
                   str_replace(dirname(rstudioapi::getSourceEditorContext()$path),
                               'EDA', 'Article Collection'),
                   "articles_zenodo.db")

con <- dbConnect(SQLite(), dbname=db.file)
lean <- dbGetQuery(con,'select * from lean')

#Plot the bias
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

#Plot hyperpartisan
lean %>%
    mutate(hyperpartisan = factor(hyperpartisan, levels = c('true', 'false'))) %>%
    ggplot() +
    geom_bar(aes(x=hyperpartisan, fill=hyperpartisan), stat = 'count') +
    geom_text(stat='count',
              aes(x=hyperpartisan, label=scales::comma(..count..)),
              vjust=1,
              color=c('white', 'white')) +
    scale_fill_manual(values = c('olivedrab3', 'grey30')) +
    theme_grey() +
    theme(legend.position = 'none') +
    labs(x='Article Hyperpartisan', y='# of Articles') +
    scale_y_continuous(labels = scales::comma)
