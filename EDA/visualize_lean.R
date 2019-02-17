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

#Most frequent sources
lean %>%
    mutate(source = str_split(str_split(url, '//', simplify = T)[,2],
                              fixed('.'), simplify = T)[,1],
           bias = factor(bias,
                         levels = c('left', 'left-center',
                                    'least',
                                    'right-center', 'right'))) %>%
    group_by(source, bias) %>%
    count() %>%
    arrange(desc(n)) %>%
    head(30) %>%
    ggplot(aes(x= reorder(source, -n), y=n, fill=bias)) +
    geom_bar(stat = 'identity') +
    scale_fill_manual(values = c('dodgerblue2', 'lightblue',
                                 'gray',
                                 'indianred1', 'firebrick')) +
    theme_grey() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = c(0.8, 0.6),
          legend.background = element_rect(color = 'grey30')) +
    labs(x=NULL, y='# of Articles')
    



