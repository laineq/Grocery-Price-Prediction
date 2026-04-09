library(shiny)
library(dplyr)
library(ggplot2)
library(readr)
library(tidyr)

# =========================
# LOAD DATA
# =========================
price_df <- read_csv("avocado_price_adjusted.csv")
gas_df <- read_csv("gas_price.csv")
xrate_df <- read_csv("xrate_adjusted.csv")
import_df <- read_csv("avocado_import.csv")
weather_df <- read_csv("mexico_weather_adjusted.csv")
lag_df <- read_csv("avocado_correlations_lag12.csv")

avocado_df <- read_csv("avocado_prediction.csv")

tomato_df <- read_csv("tomato_prediction.csv")
tomato_price_df <- read_csv("tomato_price_adjusted.csv")
tomato_import_df <- read_csv("tomato_import.csv")
tomato_lag_df <- read_csv("tomato_correlations_lag12.csv")

# =========================
# DATA PREP
# =========================
to_date <- function(df) {
  df %>% mutate(date = as.Date(paste0(date, "-01"))) %>% arrange(date)
}

price_df <- to_date(price_df)
gas_df <- to_date(gas_df)
xrate_df <- to_date(xrate_df)
import_df <- to_date(import_df)
avocado_df <- to_date(avocado_df)

tomato_df <- to_date(tomato_df)
tomato_price_df <- to_date(tomato_price_df)
tomato_import_df <- to_date(tomato_import_df)

weather_df <- weather_df %>%
  filter(STATE %in% c("Jalisco", "Estado de México", "Michoacán")) %>%
  mutate(date = as.Date(paste0(date, "-01"))) %>%
  group_by(date) %>%
  summarise(
    MEAN_C = mean(MEAN_C, na.rm = TRUE),
    PRECIPITATION_MM = mean(PRECIPITATION_MM, na.rm = TRUE)
  )

normalize <- function(x) {
  (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
}

# =========================
# UI
# =========================
ui <- fluidPage(
  
  # ✅ MOBILE FIXES
  tags$head(
    tags$style(HTML("
      @media (max-width: 768px) {
        .btn {
          width: 100%;
          margin-bottom: 6px;
        }
        #overlay_plot, #tomato_overlay {
          height: 250px !important;
        }
      }
    "))
  ),
  
  titlePanel("🥑🍅 Produce Price Analysis"),
  
  tabsetPanel(
    
    tabPanel("🥑 Avocado",
             
             plotOutput("price_plot", height = "350px"),
             
             h4("Correlation Between Price and Drivers"),
             plotOutput("correlation_heatmap", height = "350px"),
             
             h4("Compare Price with Drivers"),
             fluidRow(
               actionButton("weather_temp", "🌡️ Temperature"),
               actionButton("weather_rain", "🌧️ Precipitation"),
               actionButton("import", "📦 Import Quantity"),
               actionButton("xrate_usd", "USD"),
               actionButton("xrate_mxn", "MXN"),
               actionButton("gas", "⛽ Gas")
             ),
             plotOutput("overlay_plot", height = "300px"),  # ✅ reduced
             
             h4("Lag Effect Analysis"),
             plotOutput("lag_acf_plot", height = "600px"),
             
             h4("Prediction Comparison"),
             fluidRow(
               actionButton("model_naive", "Baseline"),
               actionButton("model_sarima", "SARIMA"),
               actionButton("model_sarimax", "SARIMAX"),
               actionButton("model_xgb", "XGBoost")
             ),
             plotOutput("avocado_plot")
    ),
    
    tabPanel("🍅 Tomato",
             
             plotOutput("tomato_price_plot", height = "350px"),
             
             h4("Correlation Between Price and Drivers"),
             plotOutput("tomato_corr", height = "350px"),
             
             h4("Compare Price with Drivers"),
             fluidRow(
               actionButton("t_weather_temp", "🌡️ Temperature"),
               actionButton("t_weather_rain", "🌧️ Precipitation"),
               actionButton("t_import", "📦 Import Quantity"),
               actionButton("t_xrate_usd", "USD"),
               actionButton("t_xrate_mxn", "MXN"),
               actionButton("t_gas", "⛽ Gas")
             ),
             plotOutput("tomato_overlay", height = "300px"),  # ✅ reduced
             
             h4("Lag Effect Analysis"),
             plotOutput("tomato_lag", height = "600px"),
             
             h4("Prediction Comparison"),
             fluidRow(
               actionButton("t_model_naive", "Baseline"),
               actionButton("t_model_sarima", "SARIMA"),
               actionButton("t_model_sarimax", "SARIMAX"),
               actionButton("t_model_xgb", "XGBoost")
             ),
             plotOutput("tomato_plot")
    )
  )
)

# =========================
# SERVER
# =========================
server <- function(input, output, session) {
  
  # ===== DRIVER STATE =====
  selected_driver <- reactiveVal("weather_temp")
  t_selected_driver <- reactiveVal("weather_temp")
  
  observeEvent(input$weather_temp, { selected_driver("weather_temp") })
  observeEvent(input$weather_rain, { selected_driver("weather_rain") })
  observeEvent(input$import, { selected_driver("import") })
  observeEvent(input$xrate_usd, { selected_driver("xrate_usd") })
  observeEvent(input$xrate_mxn, { selected_driver("xrate_mxn") })
  observeEvent(input$gas, { selected_driver("gas") })
  
  observeEvent(input$t_weather_temp, { t_selected_driver("weather_temp") })
  observeEvent(input$t_weather_rain, { t_selected_driver("weather_rain") })
  observeEvent(input$t_import, { t_selected_driver("import") })
  observeEvent(input$t_xrate_usd, { t_selected_driver("xrate_usd") })
  observeEvent(input$t_xrate_mxn, { t_selected_driver("xrate_mxn") })
  observeEvent(input$t_gas, { t_selected_driver("gas") })
  
  # ===== MODEL STATE =====
  selected_model <- reactiveVal("naive")
  t_selected_model <- reactiveVal("naive")
  
  observeEvent(input$model_naive, { selected_model("naive") })
  observeEvent(input$model_sarima, { selected_model("sarima") })
  observeEvent(input$model_sarimax, { selected_model("sarimax") })
  observeEvent(input$model_xgb, { selected_model("xgb") })
  
  observeEvent(input$t_model_naive, { t_selected_model("naive") })
  observeEvent(input$t_model_sarima, { t_selected_model("sarima") })
  observeEvent(input$t_model_sarimax, { t_selected_model("sarimax") })
  observeEvent(input$t_model_xgb, { t_selected_model("xgb") })
  
  # ===== PRICE =====
  output$price_plot <- renderPlot({
    ggplot(price_df, aes(date, price_adjusted)) +
      geom_line(color = "#568203", size = 1.2) +
      theme_minimal()
  })
  
  output$tomato_price_plot <- renderPlot({
    ggplot(tomato_price_df, aes(date, price_adjusted)) +
      geom_line(color = "#EC2D01", size = 1.2) +
      theme_minimal()
  })
  
  # ===== CORRELATION =====
  make_corr <- function(df_price, df_import) {
    df_price %>%
      select(date, price_adjusted) %>%
      left_join(gas_df, by = "date") %>%
      left_join(xrate_df, by = "date") %>%
      left_join(df_import, by = "date") %>%
      left_join(weather_df, by = "date") %>%
      rename(
        Price = price_adjusted,
        Gas = integrated_gas_price,
        USD = USD_CAD,
        MXN = MXN_CAD,
        Import = qty,
        Temp = MEAN_C,
        Rain = PRECIPITATION_MM
      )
  }
  
  output$correlation_heatmap <- renderPlot({
    df <- make_corr(price_df, import_df)
    corr <- cor(df %>% select(Price, Temp, Rain, Import, USD, MXN, Gas),
                use = "complete.obs")
    
    corr_long <- as.data.frame(as.table(corr))
    
    ggplot(corr_long, aes(Var1, Var2, fill = Freq)) +
      geom_tile() +
      geom_text(aes(label = round(Freq, 2))) +
      scale_fill_gradient2(low = "#568203", mid = "white", high = "#EC2D01") +
      theme_minimal()
  })
  
  output$tomato_corr <- renderPlot({
    df <- make_corr(tomato_price_df, tomato_import_df)
    corr <- cor(df %>% select(Price, Temp, Rain, Import, USD, MXN, Gas),
                use = "complete.obs")
    
    corr_long <- as.data.frame(as.table(corr))
    
    ggplot(corr_long, aes(Var1, Var2, fill = Freq)) +
      geom_tile() +
      geom_text(aes(label = round(Freq, 2))) +
      scale_fill_gradient2(low = "#568203", mid = "white", high = "#EC2D01") +
      theme_minimal()
  })
  
  # ===== OVERLAY =====
  make_overlay <- function(df_price, df_import, driver) {
    
    df <- df_price %>% mutate(price = price_adjusted)
    
    if (driver == "gas") {
      df <- left_join(df, gas_df, by = "date") %>%
        mutate(val = integrated_gas_price)
      
    } else if (driver == "xrate_usd") {
      df <- left_join(df, xrate_df, by = "date") %>%
        mutate(val = USD_CAD)
      
    } else if (driver == "xrate_mxn") {
      df <- left_join(df, xrate_df, by = "date") %>%
        mutate(val = MXN_CAD)
      
    } else if (driver == "import") {
      df <- left_join(df, df_import, by = "date") %>%
        mutate(val = qty)
      
    } else if (driver == "weather_temp") {
      df <- left_join(df, weather_df, by = "date") %>%
        mutate(val = MEAN_C)
      
    } else if (driver == "weather_rain") {
      df <- left_join(df, weather_df, by = "date") %>%
        mutate(val = PRECIPITATION_MM)
    }
    
    df %>%
      mutate(
        price = normalize(price),
        val = normalize(val)
      )
  }
  
  output$overlay_plot <- renderPlot({
    df <- make_overlay(price_df, import_df, selected_driver())
    
    ggplot(df, aes(date)) +
      geom_line(aes(y = price), color = "#568203", size = 1) +
      geom_line(aes(y = val), color = "#EC2D01", size = 1) +
      theme_minimal()
  })
  
  output$tomato_overlay <- renderPlot({
    
    df <- make_overlay(tomato_price_df, tomato_import_df, t_selected_driver())
    
    # ✅ DOWNsample for mobile readability
    df <- df %>%
      slice(seq(1, n(), by = 3))
    
    ggplot(df, aes(date)) +
      
      # thinner lines for dense data
      geom_line(aes(y = price), color = "#568203", size = 0.8) +
      geom_line(aes(y = val), color = "#EC2D01", size = 0.8) +
      
      # keep labels readable
      annotate("text",
               x = max(df$date),
               y = 1,
               label = "Price",
               color = "#568203",
               hjust = 1,
               vjust = -0.5,
               size = 3.5) +
      
      annotate("text",
               x = max(df$date),
               y = 0.85,
               label = "Driver",
               color = "#EC2D01",
               hjust = 1,
               size = 3.5) +
      
      theme_minimal() +
      labs(title = "Price vs Driver (Normalized)")
  })
  
  # ===== LAG (FIXED) =====
  make_lag_plot <- function(df, color) {
    
    colnames(df) <- tolower(colnames(df))
    
    lag_col <- names(df)[grepl("lag", names(df))]
    corr_col <- names(df)[grepl("cor", names(df))]
    var_col <- names(df)[grepl("var|feature|driver|name", names(df))]
    
    if (length(var_col) > 0 && length(corr_col) > 0) {
      df_long <- df %>%
        rename(
          lag = all_of(lag_col[1]),
          correlation = all_of(corr_col[1]),
          variable = all_of(var_col[1])
        )
    } else {
      df_long <- df %>%
        pivot_longer(-all_of(lag_col), names_to = "variable", values_to = "correlation") %>%
        rename(lag = all_of(lag_col))
    }
    
    df_long <- df_long %>%
      mutate(variable = case_when(
        variable == "import_qty" ~ "Import Quantity",
        variable == "integrated_gas_price" ~ "Gas",
        variable == "MEAN_C" ~ "Temperature",
        variable == "PRECIPITATION_MM" ~ "Precipitation",
        TRUE ~ variable
      ))
    
    # ✅ Find max absolute correlation per variable
    highlight_df <- df_long %>%
      group_by(variable) %>%
      filter(abs(correlation) == max(abs(correlation), na.rm = TRUE)) %>%
      ungroup()
    
    ggplot(df_long, aes(lag, correlation)) +
      
      # base ACF lines
      geom_segment(aes(xend = lag, yend = 0), color = color, size = 1) +
      geom_point(color = color, size = 2) +
      
      # ✅ highlight circle
      geom_point(data = highlight_df,
                 aes(lag, correlation),
                 color = "black",
                 size = 4,
                 shape = 21,
                 stroke = 1.5) +
      
      # ✅ label value
      geom_text(data = highlight_df,
                aes(label = round(correlation, 2)),
                vjust = -1,
                size = 3.5,
                fontface = "bold") +
      
      geom_hline(yintercept = 0, linetype = "dashed", color = color) +
      
      facet_wrap(~variable, scales = "free_y") +
      
      labs(
        title = "Lag Effect Analysis",
        x = "Lag",
        y = "Correlation"
      ) +
      
      theme_minimal()
  }
  
  output$lag_acf_plot <- renderPlot({
    make_lag_plot(lag_df, "#568203")
  })
  
  output$tomato_lag <- renderPlot({
    make_lag_plot(tomato_lag_df, "#EC2D01")
  })
  
  # ===== MODEL PLOT FUNCTION =====
  plot_model <- function(df, model, type) {
    
    pred <- switch(model,
                   "naive" = df$naive_prediction,
                   "sarima" = df$sarima_prediction,
                   "sarimax" = df$sarimax_prediction,
                   "xgb" = df$xgboost_prediction)
    
    # ✅ Explicit scale
    y_limits <- if (type == "avocado") {
      c(0.5, 3.5)
    } else {
      c(3.25, 7.5)
    }
    
    ggplot(df, aes(date)) +
      
      geom_line(aes(y = actual), color = "#568203", size = 1.2, na.rm = TRUE) +
      geom_line(aes(y = pred), color = "#EC2D01", size = 1.2, na.rm = TRUE) +
      
      # Labels (top right)
      annotate("text",
               x = max(df$date),
               y = y_limits[2],
               label = "Actual",
               color = "#568203",
               hjust = 1,
               vjust = -0.5,
               size = 4) +
      
      annotate("text",
               x = max(df$date),
               y = y_limits[2] - 0.3,
               label = "Prediction",
               color = "#EC2D01",
               hjust = 1,
               size = 4) +
      
      coord_cartesian(ylim = y_limits) +
      
      theme_minimal() +
      labs(title = "Prediction vs Actual")
  }
  output$avocado_plot <- renderPlot({
    plot_model(avocado_df, selected_model(), "avocado")
  })
  
  output$tomato_plot <- renderPlot({
    plot_model(tomato_df, t_selected_model(), "tomato")
  })
}

shinyApp(ui, server)