library(DBI)
library(RSQLite)
library(tidyverse)

db.file <- sprintf("%s/%s",
                   str_replace(dirname(rstudioapi::getSourceEditorContext()$path),
                               'EDA', 'Article Collection'),
                   "articles_zenodo.db")
con <- dbConnect(SQLite(), dbname=db.file)

#Political Lean----
lean <- dbGetQuery(con,'select * from lean')
bias.levels <- c('left', 'left-center', 'least', 'right-center', 'right')
#Plot the bias
lean %>%
    mutate(bias = factor(bias, levels = bias.levels)) %>%
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
           bias = factor(bias, levels = bias.levels)) %>%
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
    

#Table counting bias vs hyperpartisan
lean %>%
    mutate(bias = factor(bias, levels = bias.levels)) %>%
    group_by(hyperpartisan, bias) %>%
    count() %>%
    spread(bias, n) %>%
    knitr::kable(.)


#Content----
content <- dbGetQuery(con,'select * from content')

content %>%
    select(id, `published-at`) %>%
    mutate(Year = lubridate::year(`published-at`),
           Year = ifelse(is.na(Year), 2021, Year),
           PublishDate = ifelse(Year == 2021, 'No', 'Yes')) %>%
    group_by(Year, PublishDate) %>%
    count(.) %>%
    ggplot() +
    geom_bar(aes(x=Year, y=n, fill=PublishDate), stat = 'identity') +
    scale_y_continuous(labels = scales::comma) +
    labs(y='# Articles', x='Year Article Published') +
    theme_gray()




left_join(content %>% select(id, `published-at`),
          lean %>% select(id, bias),
          by='id') %>%
    mutate(Year = lubridate::year(`published-at`),
           Year = ifelse(is.na(Year), 2021, Year),
           PublishDate = ifelse(Year == 2021, 'No', 'Yes'),
           bias = factor(bias, levels = bias.levels)) %>%
    group_by(Year, bias) %>%
    count(.) %>%
    rename('Number of Articles' = n) %>%
    ggplot() +
    geom_tile(aes(x=Year, y=bias, fill=`Number of Articles`)) +
    geom_hline(yintercept = seq(0.5, 5.5, 1)) +
    theme_grey() +
    labs(x='Year Article Published', y='Leaning') +
    geom_vline(xintercept = 2012.5, color='red') +
    geom_text(aes(x=2011.5, y='least',
                  label='Cut Off Year (2012)'),
              angle=90, color='red')