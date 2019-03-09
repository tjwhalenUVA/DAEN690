library(DBI)
library(RSQLite)
library(tidyverse)
library(gridExtra)
library(rstudioapi)
library(scales)

plots.folder <- sprintf("%s/%s/%s", getwd(), 'EDA', "train_plots")

#Get Data from Database
db.file <- sprintf("%s/%s/%s", getwd(), 'Article Collection', "articles_zenodo.db")
con <- dbConnect(SQLite(), dbname=db.file)
lean <- dbGetQuery(con,'select * from train_lean')
content <- dbGetQuery(con,'select * from train_content')
mbfc <- dbGetQuery(con,'select * from mbfc_lean')
dbDisconnect(con)

bias.levels <- c('left', 'left-center', 'least', 'right-center', 'right')

#FULL DATA----
# Count of Articles in Leaning Categories
bias.cnt <-
    lean %>%
    mutate(bias = factor(bias, levels = bias.levels)) %>%
    group_by(bias) %>%
    count() %>%
    ggplot() +
    geom_bar(aes(x=bias, y=n, fill=bias), stat = 'identity') +
    geom_label(aes(x=bias, label=comma(n), y=n/2), size=6) +
    scale_fill_manual(values = c('dodgerblue2', 'lightblue',
                                 'gray',
                                 'indianred1', 'firebrick')) +
    theme_grey() +
    theme(legend.position = 'none') +
    labs(x='Article Leaning', y='# of Articles') +
    scale_y_continuous(labels = comma)

ggsave(filename='initial_lean_count.jpeg', plot=bias.cnt, 
       path=plots.folder, device='jpeg', dpi=600)

# Count of Hyperpartisan Articles
hyp.cnt <-
    lean %>%
    mutate(hyperpartisan = factor(hyperpartisan, levels = c('true', 'false'))) %>%
    group_by(hyperpartisan) %>%
    count() %>%
    ggplot() +
    geom_bar(aes(x=hyperpartisan, y=n, fill=hyperpartisan), stat = 'identity') +
    geom_label(aes(x=hyperpartisan, y=n/2, label=comma(n)), size=8) +
    scale_fill_manual(values = c('olivedrab3', 'grey30')) +
    theme_grey() +
    theme(legend.position = 'none') +
    labs(x='Article Hyperpartisan', y='# of Articles') +
    scale_y_continuous(labels = comma)

ggsave(filename='initial_hyperpartisan_count.jpeg', plot=hyp.cnt, 
       path=plots.folder, device='jpeg', dpi=600)

#Table counting bias vs hyperpartisan
# lean %>%
#     mutate(bias = factor(bias, levels = bias.levels)) %>%
#     group_by(hyperpartisan, bias) %>%
#     count() %>%
#     spread(bias, n) %>%
#     knitr::kable(.)


# 15 Most Frequent Publishers
art.bias.cnt <-
    lean %>%
    mutate(source = str_split(str_split(url, '//', simplify = T)[,2],
                              fixed('.'), simplify = T)[,1],
           bias = factor(bias, levels = bias.levels)) %>%
    group_by(source, bias) %>%
    count() %>%
    arrange(desc(n)) %>%
    head(15) %>%
    ggplot(aes(x= reorder(source, -n), y=n, fill=bias)) +
    geom_bar(stat = 'identity') +
    scale_fill_manual(values = c('dodgerblue2', 'lightblue',
                                 'gray',
                                 'indianred1', 'firebrick')) +
    theme_grey() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = c(0.8, 0.6),
          legend.background = element_rect(color = 'grey30')) +
    labs(x=NULL, y='# of Articles') +
    scale_y_continuous(labels = comma)

ggsave(filename='initial_frequent_publishers.jpeg', plot=art.bias.cnt, 
       path=plots.folder, device='jpeg', dpi=600)

# Count of Articles by Year
art.year.cnt <-
    content %>%
    left_join(., lean, by='id') %>%
    select(id, `published-at`, bias) %>%
    mutate(Year = lubridate::year(`published-at`),
           Year = ifelse(is.na(Year), 2021, Year)) %>%
    group_by(Year, bias) %>%
    count(.) %>%
    ungroup() %>%
    rename(Leaning = bias) %>%
    ggplot() +
    geom_bar(aes(x=Year, y=n, fill=Leaning), position = 'stack', stat='identity') +
    geom_vline(xintercept=2008.5, color='red') +
    geom_text(aes(x=2007.5, y=75000, label='Cut Off Year (2009)'), color='red', angle=90) +
    scale_y_continuous(labels = comma) +
    labs(y='# Articles', x='Year Article Published') +
    theme_gray() +
    scale_fill_manual(values = c('dodgerblue2', 'lightblue',
                                 'gray',
                                 'indianred1', 'firebrick')) +
    scale_x_continuous(breaks = seq(1964, 2018, 5)) +
    theme(legend.position = c(0.4, 0.6),
          legend.background = element_rect(color = 'grey30'))

ggsave(filename='initial_article_count_by_year.jpeg', plot=art.year.cnt, 
       path=plots.folder, device='jpeg', dpi=600)


# Article Leaning vs Publication Year; Filled by Article Count
bias.yr.tile <-
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
    geom_hline(yintercept = seq(0.5, 5.5, 1), color='white') +
    geom_vline(xintercept = seq(1964.5, 2018.5, 1), color='white') +
    theme_classic() +
    labs(x='Year Article Published', y='Leaning') +
    scale_x_continuous(breaks = seq(1964, 2019, 5)) +
    theme(axis.line = element_blank(),
          axis.ticks = element_blank()) +
    scale_fill_gradient(low='grey70', high='darkgreen') +
    geom_vline(xintercept = 2008.5, color='red') +
    geom_text(aes(x=2007.5, y='least',
                  label='Cut Off Year (2009)'),
              angle=90, color='red')

ggsave(filename='initial_lean_vs_year_heatmap.jpeg', plot=bias.yr.tile, 
       path=plots.folder, device='jpeg', dpi=600)


#FINAL DATA----
# Reduced Data Set
reduced.df <-
    left_join(content, lean, by='id') %>%
    mutate(Year = lubridate::year(`published-at`),
           Year = ifelse(is.na(Year), 2021, Year),
           PublishDate = ifelse(Year == 2021, 'No', 'Yes'),
           bias_final = factor(bias_final, levels = bias.levels)) %>%
    filter(!is.na(`published-at`),
           Year >= 2009,
           url_keep == 1)


# Count of Articles in Leaning Categories
red.bias.cnt <-
    reduced.df %>%
    group_by(bias_final) %>%
    count() %>%
    ggplot() +
    geom_bar(aes(x=bias_final, y=n, fill=bias_final), stat = 'identity') +
    geom_label(aes(x=bias_final, label=comma(n), y=n/2), size=6) +
    scale_fill_manual(values = c('dodgerblue2', 'lightblue',
                                 'gray',
                                 'indianred1', 'firebrick')) +
    theme_grey() +
    theme(legend.position = 'none') +
    labs(x='Article Leaning', y='# of Articles') +
    scale_y_continuous(labels = comma)

ggsave(filename='final_lean_count.jpeg', plot=red.bias.cnt, 
       path=plots.folder, device='jpeg', dpi=600)

# Count of Hyperpartisan Articles
red.hyp.cnt <-
    reduced.df %>%
    mutate(hyperpartisan = factor(hyperpartisan, levels = c('true', 'false'))) %>%
    group_by(hyperpartisan) %>%
    count() %>%
    ggplot() +
    geom_bar(aes(x=hyperpartisan, y=n, fill=hyperpartisan), stat = 'identity') +
    geom_label(aes(x=hyperpartisan, y=n/2, label=comma(n)), size=8) +
    scale_fill_manual(values = c('olivedrab3', 'grey30')) +
    theme_grey() +
    theme(legend.position = 'none') +
    labs(x='Article Hyperpartisan', y='# of Articles') +
    scale_y_continuous(labels = scales::comma)

ggsave(filename='final_hyperpartisan_count.jpeg', plot=red.hyp.cnt, 
       path=plots.folder, device='jpeg', dpi=600)

#Table counting bias vs hyperpartisan
# reduced.df %>%
#     mutate(bias = factor(bias, levels = bias.levels)) %>%
#     group_by(hyperpartisan, bias) %>%
#     count() %>%
#     spread(bias, n) %>%
#     knitr::kable(.)


# 15 Most Frequent Publishers
red.art.bias.cnt <-
    reduced.df %>%
    mutate(source = str_split(str_split(url, '//', simplify = T)[,2],
                              fixed('.'), simplify = T)[,1]) %>%
    group_by(source, bias_final) %>%
    count() %>%
    ungroup() %>%
    arrange(desc(n)) %>%
    head(15) %>%
    rename(Leaning = bias_final) %>%
    ggplot(aes(x= reorder(source, -n), y=n, fill=Leaning)) +
    geom_bar(stat = 'identity') +
    scale_fill_manual(values = c('dodgerblue2', 'lightblue',
                                 'gray',
                                 'indianred1', 'firebrick')) +
    theme_grey() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = c(0.8, 0.6),
          legend.background = element_rect(color = 'grey30')) +
    labs(x=NULL, y='# of Articles') +
    scale_y_continuous(labels = scales::comma)

ggsave(filename='final_frequent_publishers.jpeg', plot=red.art.bias.cnt, 
       path=plots.folder, device='jpeg', dpi=600)

# Count of Articles by Year
red.art.year.cnt <-
    reduced.df %>%
    select(id, `published-at`, bias_final) %>%
    mutate(Year = lubridate::year(`published-at`),
           Year = ifelse(is.na(Year), 2021, Year)) %>%
    group_by(Year, bias_final) %>%
    count(.) %>%
    ungroup() %>%
    rename(Leaning = bias_final) %>%
    ggplot() +
    geom_bar(aes(x=Year, y=n, fill=Leaning), position = 'stack', stat='identity') +
    geom_vline(xintercept=2008.5, color='red') +
    geom_text(aes(x=2008, y=75000, label='Cut Off Year (2009)'), color='red', angle=90) +
    scale_y_continuous(labels = scales::comma) +
    labs(y='# Articles', x='Year Article Published') +
    theme_gray() +
    scale_fill_manual(values = c('dodgerblue2', 'lightblue',
                                 'gray',
                                 'indianred1', 'firebrick')) +
    scale_x_continuous(breaks = seq(2009, 2018, 2)) +
    theme(legend.position = c(0.3, 0.7),
          legend.background = element_rect(color = 'grey30'))

ggsave(filename='final_article_count_by_year.jpeg', plot=red.art.year.cnt, 
       path=plots.folder, device='jpeg', dpi=600)


# Article Leaning vs Publication Year; Filled by Article Count
red.bias.yr.tile <-
    reduced.df %>%
    mutate(Year = lubridate::year(`published-at`),
           Year = ifelse(is.na(Year), 2021, Year),
           PublishDate = ifelse(Year == 2021, 'No', 'Yes'),
           bias = factor(bias, levels = bias.levels)) %>%
    group_by(Year, bias_final) %>%
    count(.) %>%
    rename('Number of Articles' = n) %>%
    ggplot() +
    geom_tile(aes(x=Year, y=bias_final, fill=`Number of Articles`)) +
    geom_hline(yintercept = seq(0.5, 5.5, 1), color='white') +
    geom_vline(xintercept = seq(2008.5, 2018.5, 1), color='white') +
    theme_classic() +
    labs(x='Year Article Published', y='Leaning') +
    scale_x_continuous(breaks = seq(2009, 2019, 2)) +
    theme(axis.line = element_blank(),
          axis.ticks = element_blank()) +
    scale_fill_gradient(low='grey70', high='darkgreen')

ggsave(filename='final_lean_vs_year_heatmap.jpeg', plot=red.bias.yr.tile, 
       path=plots.folder, device='jpeg', dpi=600)


#Media Bias Fact Check----
bias.levels <- c('left', 'leftcenter', 'least', 'rightcenter', 'right')
#Extract URL main and join mbfc to zenodo
lean.df <-
    lean %>%
    mutate(source = str_split(str_split(url, '//', simplify = T)[,2], fixed('.'), simplify = T)[,1],
           bias = str_replace(bias, '-', '')) %>%
    select(source, zenodo = bias) %>%
    group_by(source, zenodo) %>%
    count(.) %>%
    select(-n) %>%
    left_join(.,
              mbfc %>%
                  mutate(source = str_replace(sourceURL, fixed('www.'), ''),
                         source = str_split(source, fixed('//'), simplify = T)[,2],
                         source = str_split(source, fixed('.'), simplify = T)[,1]) %>%
                  select(source, mbfc = bias),
              by='source') %>%
    ungroup() %>%
    mutate(zenodo = factor(zenodo, levels = bias.levels),
           mbfc = ifelse(mbfc == 'center', 'least', mbfc),
           mbfc = factor(mbfc, levels = bias.levels))

#Line aesthetics for plot
dividers.x <- seq(0.5, 5.5, 1)
dividers.y <- seq(0.5, 5.5, 1)
grid.df <-
    bind_rows(data.frame(x = dividers.x, xend = dividers.x, y=0.5, yend=5.5),
              data.frame(x = 0.5, xend = 5.5, y=dividers.y, yend=dividers.y))
#Similaruty counts for plot
sim.df <-
    lean.df %>%
    group_by(mbfc, zenodo) %>%
    count(.)
#Plot confusion matrix
mbfc.plot <-
    lean.df %>%
    mutate(match = ifelse(mbfc == zenodo, 'yes', 'no')) %>%
    ggplot() +
    geom_point(aes(x=mbfc, y=zenodo, color=match), position = 'jitter', show.legend = F) +
    geom_text(data = sim.df, aes(x=mbfc, y=zenodo, label=n), size = 8) +
    geom_segment(data = grid.df, aes(x=x, y=y, xend=xend, yend=yend)) +
    theme_classic() +
    theme(axis.line = element_blank(),
          axis.ticks = element_blank()) +
    scale_color_manual(values = c('grey30', 'purple', 'grey80')) +
    labs(x='mediabiasfactcheck.com Leaning', y='Zenodo Hyperpartisan\nDataset Leaning')

ggsave(filename='mbfc_vs_zenodo.jpeg', plot=mbfc.plot, 
       path=plots.folder, device='jpeg', dpi=600)

disagree.df <-
    lean.df %>%
    mutate(match = ifelse(mbfc == zenodo, 'yes', 'no')) %>%
    filter(match == 'no' | is.na(match))