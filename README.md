# **Synoptic Meteorology Analyzer**  

## **📌 Visão Geral**  

Este projeto é um **Sistema de Análise Sinótica Automatizada** desenvolvido em Python para processar e visualizar dados meteorológicos em escala regional, com foco no território brasileiro. Ele permite:  

- **Coletar dados meteorológicos** de múltiplas cidades (via API OpenWeatherMap ou dados simulados).  
- **Processar e interpolar** campos meteorológicos (pressão, temperatura, vento).  
- **Identificar sistemas de pressão** (altas e baixas pressões).  
- **Gerar cartas sinóticas** (mapas meteorológicos profissionais).  
- **Produzir relatórios automáticos** com análise das condições climáticas.  

É uma ferramenta útil para **meteorologistas, estudantes e entusiastas** que desejam automatizar a análise do tempo em diferentes regiões.  

---

## **🔧 Funcionalidades Detalhadas**  

### **1. Coleta de Dados Meteorológicos**  
- **Fonte de dados**:  
  - **OpenWeatherMap API** (requer chave de acesso).  
  - **Modo simulado** (gera dados aleatórios para testes).  
- **Variáveis coletadas**:  
  - Temperatura (°C)  
  - Pressão atmosférica (hPa)  
  - Umidade relativa (%)  
  - Velocidade e direção do vento (m/s, graus)  

### **2. Processamento de Dados**  
- **Interpolação espacial**:  
  - Converte dados pontuais (estações meteorológicas) em campos contínuos usando interpolação linear.  
  - Gera grades regulares para visualização em mapas.  
- **Detecção de sistemas de pressão**:  
  - Identifica **centros de alta pressão (A)** e **baixa pressão (B)** automaticamente.  
  - Filtra mínimos e máximos locais para evitar falsos positivos.  

### **3. Visualização com Cartas Sinóticas**  
- **Mapa base**:  
  - Limites geopolíticos (estados, países, costa).  
  - Relevo simplificado (terra e oceano).  
- **Elementos meteorológicos plotados**:  
  - **Isóbaras** (linhas de pressão atmosférica).  
  - **Campos de temperatura** (preenchimento colorido).  
  - **Direção do vento** (setas indicativas).  
  - **Posição das estações meteorológicas** (pontos pretos).  
  - **Centros de pressão** (A = Alta, B = Baixa).  

### **4. Relatório Meteorológico Automático**  
- **Estatísticas gerais**:  
  - Médias de temperatura, pressão, umidade e vento.  
- **Análise de sistemas de pressão**:  
  - Quantidade, posição e intensidade de altas/baixas pressões.  
- **Condições por cidade**:  
  - Detalhamento das variáveis meteorológicas em cada local.  

---

## **📦 Dependências**  

| Biblioteca | Função |
|------------|--------|
| `requests` | Requisições HTTP à API OpenWeatherMap |
| `pandas` | Manipulação de dados em DataFrames |
| `numpy` | Cálculos numéricos e álgebra linear |
| `matplotlib` | Geração de gráficos e mapas |
| `cartopy` | Visualização geográfica (projeções, mapas) |
| `scipy` | Interpolação e filtragem de dados |
| `datetime` | Manipulação de datas e horários |

---

## **🚀 Como Usar**  

### **1. Configuração Inicial**  
- Instale as dependências:  
  ```bash
  pip install requests pandas numpy matplotlib cartopy scipy
  ```
- Para usar a **API OpenWeatherMap**, insira sua chave:  
  ```python
  analyzer = SynopticMeteorologyAnalyzer(api_key='SUA_CHAVE_API')
  ```
- **Sem chave API?** O sistema usará **dados simulados** automaticamente.  

### **2. Execução**  
Execute o script principal (`main.py` ou o arquivo correspondente). Os resultados serão salvos em:  
- **`resultados/carta_sinotica.png`** → Mapa meteorológico.  
- **`resultados/relatorio_meteorologico.txt`** → Análise textual.  

### **3. Personalização**  
- **Adicionar/remover cidades**:  
  Edite a lista em `get_weather_data()`.  
- **Ajustar parâmetros de interpolação**:  
  Modifique `grid_size` em `interpolate_field()`.  
- **Alter estilo do mapa**:  
  Personalize cores e elementos em `create_synoptic_chart()`.  

---

## **📊 Exemplo de Saída**  

### **Carta Sinótica**  
![Exemplo de Carta Sinótica](resultados/carta_sinotica.png)  

### **Relatório Meteorológico**  
```plaintext
RELATÓRIO METEOROLÓGICO - 2023-11-15 14:30:00  
============================================  

CONDIÇÕES GERAIS:  
- Temperatura Média: 24.8 °C  
- Pressão Média: 1013.5 hPa  
- Umidade Média: 72.1%  
- Vento Médio: 5.3 m/s  

SISTEMAS DE PRESSÃO IDENTIFICADOS (2):  
1. ALTA PRESSÃO (A)  
   - Local: 12.5°S, 45.2°W  
   - Pressão: 1028 hPa  
2. BAIXA PRESSÃO (B)  
   - Local: 22.1°S, 50.8°W  
   - Pressão: 1002 hPa  

CONDIÇÕES POR CIDADE:  
► São Paulo:  
   - Temp: 22.1°C | Pressão: 1015 hPa  
   - Vento: 4.2 m/s (120°)  
► Rio de Janeiro:  
   - Temp: 26.5°C | Pressão: 1012 hPa  
   - Vento: 6.1 m/s (80°)  
...
```

---

## **📌 Aplicações**  
✔ **Previsão do tempo simplificada**  
✔ **Estudos em meteorologia sinótica**  
✔ **Aulas e demonstrações em climatologia**  
✔ **Análise de padrões atmosféricos regionais**  

---

## **📜 Licença**  
Este projeto é open-source (MIT). Sinta-se à vontade para **contribuir, modificar e distribuir**!  

🔗 **GitHub**: [SeuRepositório](https://github.com/brjatoba92/met_sin_python)  
📧 **Contato**: [E-mail](brunojatobadev@gmail.com)  

--- 

**🌟 Dúvidas? Sugestões? Abra uma *issue* ou contribua!**

---

# Detecção e Reastreador de Sistemas Meteorológicos