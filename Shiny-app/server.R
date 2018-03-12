library(shiny)
library(keras)
library(flexdashboard)

source("Data/prediction.R")

s = c(0, 0.4)
w = c(0.4, 0.8)
d = c(0.8, 1)

server <- function(input, output){
  
  current_input <- function(){
    input$comment
  }
  
  predictions <- reactive({
    input_text <- current_input()
    preds <- comment_preds(input_text)
  })
  

  output$Toxic <- renderGauge({ 
    preds <- predictions()
    toxic <- max(0, preds[,"toxic"])
    gauge(round(toxic, 5), min = 0, max = 1, sectors = gaugeSectors(success = s, warning = w, danger = d))
  })
  output$Severe_Toxic <- renderGauge({
    preds <- predictions()
    severe_toxic <- max(0, preds[,'severe_toxic'])
    gauge(round(severe_toxic, 5), min = 0, max = 1, sectors = gaugeSectors(success = s, warning = w, danger = d))
  })
  output$Obscene <- renderGauge({
    preds <- predictions()
    obscene <- max(0, preds[,"obscene"])
    gauge(round(obscene, 5), min = 0, max = 1, sectors = gaugeSectors(success = s, warning = w, danger = d))
  })
  output$Insult <- renderGauge({
    preds <- predictions()
    insult <- max(0, preds[,"insult"])
    gauge(round(insult, 5), min = 0, max = 1, sectors = gaugeSectors(success = s, warning = w, danger = d))
  })
  output$Threat <- renderGauge({
    preds <- predictions()
    threat <- max(0, preds[,"threat"])
    gauge(round(threat, 5), min = 0, max = 1, sectors = gaugeSectors(success = s, warning = w, danger = d))
  })
  output$Identity_Hate <- renderGauge({
    preds <- predictions()
    identity_hate <- max(0, preds[,"identity_hate"])
    gauge(round(identity_hate, 5), min = 0, max = 1, sectors = gaugeSectors(success = s, warning = w, danger = d))
  })
  
 
}