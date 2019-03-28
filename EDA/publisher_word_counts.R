library(DBI)
library(RSQLite)
library(tidyverse)
library(gridExtra)
library(scales)
library(magrittr)
library(tm)

#Get Data from Database
db.file <- sprintf("%s/%s/%s", getwd(), 'Article Collection', "articles_zenodo.db")
con <- dbConnect(SQLite(), dbname=db.file)
wc.train <- dbGetQuery(con,'select * from train_publisher_WC')
wc.test <- dbGetQuery(con,'select * from test_publisher_WC')
dbDisconnect(con)


#Train----
# df.wc.train <-
#     wc.train %>%
#     mutate(word = removePunctuation(word),
#            word = removeNumbers(word)) %>%
#     filter(!(word %in% c(stopwords(), '’', '“', '”', '—', '–')),
#            word != "") %>%
#     group_by(word) %>%
#     summarise_all(., function(x) sum(x, na.rm = TRUE))

#Word Count Distribution
wc.train %>%
    ggplot() +
    geom_density(aes(x=all), fill='darkorange') +
    scale_x_log10(labels=comma) +
    labs(x='Word Count Across All Documents',
         title = 'All Unique Words in Dataset') +
    theme_classic()

wc.train %>%
    filter(!(word %in% c('s', 't', 'nt')),
           all >= 10000) %>%
    ggplot() +
    geom_density(aes(x=all), fill='darkorange') +
    scale_x_log10(labels=comma) +
    labs(x='Word Count Across All Documents',
         title = "Words Appearing >= 10,000 Times") +
    theme_classic()


#Words Occuring 10,000x or More
df.wc.high <-
    wc.train %>%
    filter(all >= 2500) %>%
    mutate(word = removePunctuation(word),
           word = removeNumbers(word)) %>%
    filter(!(word %in% c(stopwords(),
                         '’', '“', '”', '—', '–', '‘', '…',
                         's', 't', 'nt')),
           word != "") %>%
    group_by(word) %>%
    summarise_all(., function(x) sum(x, na.rm = TRUE)) %>%
    gather(publisher, pwc, -word, -all) %>%
    mutate(publisher_percent = pwc / all) %>%
    filter(publisher_percent >= 0.5)

#Line Plots
# df.wc.high %>%
#     rename('Percent Contribution' = publisher_percent) %>%
#     filter(publisher %in% c('foxbusiness', 'apnews', 'reuters')) %>%
#     ggplot(aes(x=reorder(word, -`Percent Contribution`), y=`Percent Contribution`,
#                group=publisher)) +
#     geom_line() +
#     geom_point() +
#     facet_wrap(~publisher, scales = 'free_x', nrow = 3) +
#     theme_bw() +
#     theme(axis.text.x = element_text(angle = 20)) +
#     scale_y_continuous(labels = percent) +
#     labs(x='Word')


#Heat Map
num.words <- nrow(df.wc.high)
num.pubs <- length(unique(df.wc.high$publisher))

df.wc.high %>%
    rename('Percent Contribution' = publisher_percent) %>%
    ggplot(aes(x=reorder(word, `Percent Contribution`),
               y=publisher, fill=`Percent Contribution`)) +
    geom_tile() +
    theme_classic() +
    theme(axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          panel.grid.major = element_blank(),
          axis.ticks.y = element_blank(),
          axis.line = element_blank()) +
    scale_fill_gradient(low = 'grey', high = 'red') +
    labs(title = 'Percent an Individual Publisher\nContributes to Overall Word Count in Dataset', 
         subtitle = 'Word Counts > 2,500 in Entire Dataset',
         x='Word', y='Article Source') +
    geom_hline(yintercept = seq(0.5, num.pubs+0.5, 1)) +
    geom_segment(aes(x=0.5, xend=0.5, y=0.5, yend=num.pubs+0.5)) +
    geom_segment(aes(x=num.words+0.5, xend=num.words+0.5,
                     y=0.5, yend=num.pubs+0.5))

df.wc.high %>%
    ungroup() %>%
    group_by(publisher) %>%
    count(.) %>%
    arrange(desc(n))

#Test----
df.wc.test <-
    wc.test %>%
    mutate(word = removePunctuation(word),
           word = removeNumbers(word)) %>%
    filter(!(word %in% c(stopwords(), '’', '“', '”', '—', '–')),
           word != "") %>%
    group_by(word) %>%
    summarise_all(., function(x) sum(x, na.rm = TRUE))

#Word Count Distribution
wc.test %>%
    ggplot() +
    geom_density(aes(x=all), fill='darkorange') +
    scale_x_log10(labels=comma) +
    labs(x='Word Count Across All Documents',
         title = 'All Unique Words in Dataset') +
    theme_classic()

wc.test %>%
    filter(!(word %in% c('s', 't', 'nt')),
           all >= 10000) %>%
    ggplot() +
    geom_density(aes(x=all), fill='darkorange') +
    scale_x_log10(labels=comma) +
    labs(x='Word Count Across All Documents',
         title = "Words Appearing >= 10,000 Times") +
    theme_classic()


#Words Occuring 10,000x or More
df.wc.high <-
    wc.test %>%
    filter(all >= 5000) %>%
    mutate(word = removePunctuation(word),
           word = removeNumbers(word)) %>%
    filter(!(word %in% c(stopwords(),
                         '’', '“', '”', '—', '–', '‘', '…',
                         's', 't', 'nt')),
           word != "") %>%
    group_by(word) %>%
    summarise_all(., function(x) sum(x, na.rm = TRUE)) %>%
    gather(publisher, pwc, -word, -all) %>%
    mutate(publisher_percent = pwc / all) %>%
    filter(publisher_percent >= 0.5)

#Line Plots
# df.wc.high %>%
#     rename('Percent Contribution' = publisher_percent) %>%
#     filter(publisher %in% c('foxbusiness', 'apnews', 'reuters')) %>%
#     ggplot(aes(x=reorder(word, -`Percent Contribution`), y=`Percent Contribution`,
#                group=publisher)) +
#     geom_line() +
#     geom_point() +
#     facet_wrap(~publisher, scales = 'free_x', nrow = 3) +
#     theme_bw() +
#     theme(axis.text.x = element_text(angle = 20)) +
#     scale_y_continuous(labels = percent) +
#     labs(x='Word')


#Heat Map
num.words <- nrow(df.wc.high)
num.pubs <- length(unique(df.wc.high$publisher))

df.wc.high %>%
    rename('Percent Contribution' = publisher_percent) %>%
    ggplot(aes(x=reorder(word, `Percent Contribution`),
               y=publisher, fill=`Percent Contribution`)) +
    geom_tile() +
    theme_classic() +
    theme(axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          panel.grid.major = element_blank(),
          axis.ticks.y = element_blank(),
          axis.line = element_blank()) +
    scale_fill_gradient(low = 'grey', high = 'red') +
    labs(title = 'Percent an Individual Publisher\nContributes to Overall Word Count in Dataset', 
         subtitle = 'Word Counts > 10,000 in Entire Dataset',
         x='Word', y='Article Source') +
    geom_hline(yintercept = seq(0.5, num.pubs+0.5, 1)) +
    geom_segment(aes(x=0.5, xend=0.5, y=0.5, yend=num.pubs+0.5)) +
    geom_segment(aes(x=num.words+0.5, xend=num.words+0.5,
                     y=0.5, yend=num.pubs+0.5))





#Test vs Train----
df.tvt <- 
    left_join(wc.test %>% select(word, 'test' = all),
              wc.train %>% select(word, 'train' = all)) %>%
    filter(!(word %in% c(stopwords(), '’', '“', '”', '—', '–')),
           word != "") %>%
    group_by(word) %>%
    summarise_all(., function(x) sum(x, na.rm = TRUE))

df.tvt %>%
    mutate(TrainOnly = ifelse((train > 0) & (test == 0), 1, 0),
           TestOnly = ifelse((test > 0) & (train == 0), 1, 0)) %>%
    filter(TestOnly == 1) %>%
    arrange(desc(test))


#Publisher Similarity----
df.all <-
    left_join(wc.test %>% rename('test_all' = all),
              wc.train %>% rename('train_all' = all))
