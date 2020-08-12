if (!require(pacman)) {install.packages("pacman")}
pacman::p_load('corrplot', 'tidyverse')

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

reliability_selection <- read.csv('reliability_arrays/reliability_selection_bysite.csv')
rdm_reliabilities <- reliability_selection %>% filter(grepl('rdm', instance))
image_reliabilities <- reliability_selection %>% filter(grepl('image', instance))

image_reliabilities_summary <- image_reliabilities %>% 
  group_by(cutoff, area, layer, neural_site, instance) %>% summarise(splithalf_r = mean(splithalf_r))

ggplot(rdm_reliabilities, aes(cutoff, splithalf_r, color=neural_site)) + 
         stat_summary(fun.y = mean, geom="smooth") + theme_classic() 

ggplot(rdm_reliabilities, aes(cutoff, splithalf_r, group=iteration)) + 
  facet_grid(layer ~ area) + geom_smooth(se=FALSE) + theme_bw()

ggplot(image_reliabilities_summary, aes(cutoff, splithalf_r, color=instance, group=instance)) + 
  facet_grid(layer ~ area) + geom_smooth(se=FALSE) + scale_color_discrete(guide = FALSE)

ggplot(rdm_reliabilities, aes(cutoff, splithalf_r, group=iteration)) + 
  facet_grid(layer ~ area) + geom_smooth(se=FALSE) + 
  theme_bw() + xlab('\n Cutoff') + ylab('Splithalf-R \n') +
  scale_x_continuous(breaks = c(0.25,0.5,0.75,1.0)) +
  guides(x = guide_axis(angle=45)) +
  theme(text = element_text(size=24)) +
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank()) + 
  theme(legend.background=element_blank())
  
ggplot(image_reliabilities_summary, aes(cutoff, splithalf_r, color=instance, group=instance)) + 
  facet_grid(layer ~ area) + geom_smooth(se=FALSE) + scale_color_discrete(guide = FALSE) +
  theme_bw() +  xlab('\n Cutoff') + ylab('Splithalf-R \n') +
  scale_x_continuous(breaks = c(0.25,0.5,0.75,1.0)) +
  guides(x = guide_axis(angle=45)) +
  theme(text = element_text(size=24)) +
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank()) + 
  theme(legend.background=element_blank())

brain_uber_rdm <- read.csv('brain_uber_rdm.csv') %>% 
  rename_all(list(~ stringr::str_replace_all(.,'_layer', '-')))

corrplot(cor(brain_uber_rdm), tl.col='black', tl.srt = 45, tl.cex = 1.5, cl.cex = 1.5)

