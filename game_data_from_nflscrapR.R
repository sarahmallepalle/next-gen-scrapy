library(nflscrapR)
library(dplyr)

games_2017_reg <- scrape_game_ids(2017, type = "reg")
games_2017_post <- scrape_game_ids(2017, type = "post")
games_2018_reg <- scrape_game_ids(2018, type = "reg")
games_2017_2018 <- as.data.frame(do.call("rbind", list(games_2017_reg, 
                                                       games_2017_post, 
                                                       games_2018_reg)))
pass_locations <- read.csv("pass_locations.csv")
pass_locations$game_id <- as.character(pass_locations$game_id)

pass_and_game_data <- full_join(pass_locations, games_2017_2018)
pass_and_game_data <- pass_and_game_data[which(pass_and_game_data$state_of_game != "PRE"),]
pass_and_game_data$state_of_game <- NULL
write.csv(pass_and_game_data, "./pass_and_game_data.csv")
