# Python aplicado a Meteorologia Sinótica

## Sistema de Análise de Cartas Sinóticas Automatizado
    Objetivo: 
    
        Criar um sistema que baixa dados meteorológicos em tempo real e gera cartas sinóticas automáticas.

    Funcionalidades principais:

        Coleta dados de estações meteorológicas via APIs (INMET, OpenWeatherMap)
        Interpolação espacial de dados usando métodos como Kriging
        Geração automática de mapas de pressão atmosférica, temperatura e precipitação
        Identificação automática de sistemas frontais e centros de pressão
        Interface web para visualização interativa

    Bibliotecas principais: 
        requests, 
        pandas, 
        numpy, 
        matplotlib, 
        cartopy, 
        scipy, 
        plotly, 
        dash