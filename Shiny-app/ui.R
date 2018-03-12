library(shiny)

#Define UI for Toxic comments application
ui <- fluidPage(
  
  navbarPage(title = "Comment Classifier", inverse = T, windowTitle = "Comment Classifier", fluid = T, collapsible = T,
            
            # Prediction tab
            tabPanel("Predictions",
                  
                      # Main panel with output
                      mainPanel(
                        textInput(inputId = "comment", label = "Comment:", placeholder = "Text", width = "100%"),
                        br(),
                        
                        # output with columsn "Toxic", "Severe Toxic", "Obscene"
                        fluidRow(
                          
                          # "Toxic"
                          column(width = 4, 
                                 h3("Toxic"),
                                 br(),
                                 gaugeOutput(outputId = "Toxic")),
                          
                          # "Severe Toxic"
                          column(width = 4, 
                                 h3("Severe Toxic"),
                                 br(),
                                 gaugeOutput(outputId = "Severe_Toxic")),
                          
                          # "Obscene"
                          column(width = 4, 
                                 h3("Obscene"),
                                 br(),
                                 gaugeOutput(outputId = "Obscene"))
                          
                        ),
                        
                        # output with columns "Threat", "Identity threat", "Insult"
                        fluidRow(
                          
                          # "Insult"
                          column(width = 4, 
                                 h3("Insult"),
                                 br(),
                                 gaugeOutput(outputId = "Insult")),
                          
                          # "Threat"
                          column(width = 4, 
                                 h3("Threat"),
                                 br(),
                                 gaugeOutput(outputId = "Threat")),
                          
                          # "Identity Hate"
                          column(width = 4, 
                                 h3("Identity Hate"),
                                 br(),
                                 gaugeOutput(outputId = "Identity_Hate"))
                        )
                      )
                     )
            
            # More tab
            # navbarMenu("More",
            #           tabPanel("Help"))
      
  )
  
) 