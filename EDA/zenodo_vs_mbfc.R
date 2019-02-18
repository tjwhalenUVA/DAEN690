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
lean.df <-
    z %>%
    mutate(source = str_split(str_split(url, '//', simplify = T)[,2], fixed('.'), simplify = T)[,1],
           bias = str_replace(bias, '-', '')) %>%
    select(source, zenodo = bias) %>%
    group_by(source, zenodo) %>%
    count(.) %>%
    select(-n) %>%
    left_join(.,
              m %>%
                  mutate(source = str_replace(sourceURL, fixed('www.'), ''),
                         source = str_split(source, fixed('//'), simplify = T)[,2],
                         source = str_split(source, fixed('.'), simplify = T)[,1]) %>%
                  select(source, mbfc = bias),
              by='source') %>%
    ungroup() %>%
    mutate(zenodo = factor(zenodo, levels = bias.levels),
           mbfc = ifelse(mbfc == 'center', 'least', mbfc),
           mbfc = factor(mbfc, levels = bias.levels))


dividers.x <- seq(0.5, 5.5, 1)
dividers.y <- seq(0.5, 5.5, 1)
grid.df <-
    bind_rows(data.frame(x = dividers.x, xend = dividers.x, y=0.5, yend=5.5),
              data.frame(x = 0.5, xend = 5.5, y=dividers.y, yend=dividers.y))

sim.df <-
    lean.df %>%
    group_by(mbfc, zenodo) %>%
    count(.)

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
    labs(x='mediabiasfactcheck.com', y='Zenodo Hyperpartisan\nDataset')
