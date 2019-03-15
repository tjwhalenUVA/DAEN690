library(DBI)
library(RSQLite)
library(tidyverse)
library(gridExtra)
library(rstudioapi)
library(scales)
library(magrittr)
library(ggpirate)

plots.folder <- sprintf("%s/%s/%s", getwd(), 'Results', "cnn_plots")
results.file <- sprintf("%s/%s/%s", getwd(), 'Results', "cnn_grid_results_raw.xlsx")

parameters <- c("id",
                "vocabsize", "capturefraction",
                "convolutionfilters", "convolutionkernel",
                "poolsize", "denseunits",
                "dropoutfraction")
trainaccuracy <- paste('trainacc', seq(1, 15, 1), sep='_')
valaccuracy <- paste('valacc', seq(1, 15, 1), sep='_')
trainloss <- paste('trainloss', seq(1, 15, 1), sep='_')
valloss <- paste('valloss', seq(1, 15, 1), sep='_')

results <- readxl::read_xlsx(results.file, skip = 2,
                             col_names = c(parameters, trainaccuracy,
                                           valaccuracy, trainloss, valloss)) %>%
    mutate(vocabsize = as.character(vocabsize),
           capturefraction = as.character(capturefraction),
           convolutionfilters = as.character(convolutionfilters),
           convolutionfilters = factor(convolutionfilters, levels=c('5', '10', '25', '50'))) %>%
    gather(cName, value, -parameters) %>%
    mutate(metric = str_split(cName, '_', simplify = T)[,1]) %>%
    select(-cName) %>%
    group_by(.dots = c(parameters, 'metric')) %>%
    summarise(Minimum = min(value, na.rm = T),
              Maximum = max(value, na.rm = T)) %>%
    mutate(value = ifelse(grepl('acc', metric), Maximum, Minimum))


# max validation accuracy of 0.85821
# vocabSize=1000
# captureFraction=0.95
# convolutionFilters=50
# convolutionKernel=15
# convolutionActivation='relu'
# poolSize=10
# flattenLayers=True
# denseUnits=50
# denseActivation='relu'
# dropoutFraction=0.0

#Explore Vocabulary Size
results %>%
    ggplot(aes(x=vocabsize, y=value)) +
    geom_pirate(aes(color=vocabsize),
                bars = F,
                points_params = list(size = 0.5),
                lines_params = list(size=0.2),
                cis_params = list(alpha=0.01),
                jitter_width = 0.1) +
    facet_wrap(metric~.,
               scales = 'free_y', labeller = label_both,
               nrow = 2) +
    theme_bw() +
    labs(x='Vocabulary Size', y=NULL,
         title = 'Grid Search Results by Vocabulary Size',
         subtitle = 'All Grid Search Results')

#Explore Cat. Frac given Vocabulary Size 1000
results %>%
    filter(vocabsize == '1000') %>%
    ggplot(aes(x=capturefraction, y=value)) +
    geom_pirate(aes(color=capturefraction),
                bars = F,
                points_params = list(size = 0.5),
                lines_params = list(size=0.2),
                cis_params = list(alpha=0.01),
                jitter_width = 0.1) +
    facet_wrap(metric~.,
               scales = 'free_y', labeller = label_both,
               nrow = 2) +
    theme_bw() +
    labs(x='Capture Fraction', y=NULL,
         title = 'Grid Search Results by Capture Fraction',
         subtitle = 'Vocab Size 1,000')


#Explore Conv Filters given Vocabulary Size 1000 & CapFrac 0.95
results %>%
    filter(vocabsize == '1000',
           capturefraction == '0.95') %>%
    ggplot(aes(x=convolutionfilters, y=value)) +
    geom_pirate(aes(color=convolutionfilters),
                bars = F,
                points_params = list(size = 0.5),
                lines_params = list(size=0.2),
                cis_params = list(alpha=0.01),
                jitter_width = 0.1) +
    facet_wrap(metric~.,
               scales = 'free_y', labeller = label_both,
               nrow = 2) +
    theme_bw() +
    labs(x='Convolution Filters', y=NULL,
         title = 'Grid Search Results by Convolution Filters',
         subtitle = 'Vocab Size 1,000; Capture Fraction 0.95')













#Convolution Filters vs Capture Fraction vs vocab size
results %>%
    ggplot(aes(x=capturefraction, y=value)) +
    geom_pirate(aes(color=vocabsize), show.legend = T,
                bars = F,
                points_params = list(size = 0.5),
                lines_params = list(size=0.2),
                cis_params = list(alpha=0.01),
                jitter_width = 0.1) +
    facet_grid(metric~convolutionfilters,
               scales = 'free_y', labeller = label_both) +
    theme_bw() +
    labs(x='Capture Fraction', y=NULL) +
    theme(legend.position = 'bottom')

#Compare higher convofilter 25 v 50
results %>% ungroup() %>%
    mutate(convolutionfilters = as.character(convolutionfilters)) %>%
    filter(convolutionfilters %in% c('25', '50'),
           capturefraction %in% c('0.95', '0.9999'),
           vocabsize %in% c('1000', '10000')) %>%
    ggplot(aes(x=capturefraction, y=value)) +
    geom_pirate(aes(color=convolutionfilters),
                show.legend = T, bars = F,
                points_params = list(size = 0.5),
                lines_params = list(size=0.2),
                cis_params = list(alpha=0.01),
                jitter_width = 0.1) +
    facet_grid(metric~.,
               scales = 'free_y', labeller = label_both) +
    theme_bw() +
    labs(x='Capture Fraction', y=NULL) +
    theme(legend.position = 'bottom') +
    labs(title='Vocabulary Size in [1,000, 10,000]')


#Extra
results %>%
    mutate(colRed = ifelse((capturefraction == "0.95" & metric == 'valacc' & vocabsize == "1000") | 
                               (capturefraction == "0.9999" & metric == 'valacc' & vocabsize == "1000"),
                           'yes', 'no')) %>%
    ggplot(aes(x=vocabsize, y=value)) +
    geom_pirate(aes(color=colRed),
                bars = F,
                points_params = list(size = 0.5),
                lines_params = list(size=0.2),
                cis_params = list(alpha=0.01)) +
    facet_grid(metric~capturefraction,
               scales = 'free_y', labeller = label_both) +
    theme_bw() +
    labs(x='Vocabulary Size', y=NULL)


results %>%
    mutate(colRed = ifelse((capturefraction == "0.95" & vocabsize == "1000") | 
                               (capturefraction == "0.9999" & vocabsize == "1000"),
                           'yes', 'no')) %>%
    filter(colRed == 'yes') %>%
    ggplot(aes(x=vocabsize, y=value)) +
    geom_pirate(aes(color=capturefraction), show.legend = T,
                bars = F,
                points_params = list(size = 1),
                lines_params = list(size=0.2),
                cis_params = list(alpha=0.01)) +
    facet_grid(metric~convolutionfilters,
               scales = 'free_y', labeller = label_both) +
    theme_bw() +
    labs(x='Vocabulary Size', y=NULL)
