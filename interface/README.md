# Interface Web - Sistema de Predição Lotofácil

## Visão Geral

Esta é a interface web moderna para o sistema de predição da Lotofácil, desenvolvida com HTML5, CSS3 e JavaScript vanilla. A interface oferece uma experiência completa para visualização de predições, análise de resultados e monitoramento de performance dos modelos.

## Funcionalidades Principais

### 1. Dashboard Principal
- **Predições em Tempo Real**: Visualize as predições mais recentes dos diferentes modelos
- **Histórico de Resultados**: Acompanhe o histórico completo de predições e seus resultados
- **Métricas de Performance**: Monitore a precisão e eficácia dos modelos em tempo real

### 2. Modelos Disponíveis
- **TensorFlow Básico**: Modelo de rede neural para predições baseadas em padrões históricos
- **Algoritmos Avançados**: Ensemble de algoritmos para predições mais robustas
- **Análise de Padrões**: Modelo especializado em identificação de padrões numéricos

### 3. Gráficos e Visualizações
- **Gráfico de Performance**: Acompanhe a evolução da precisão dos modelos ao longo do tempo
- **Distribuição de Números**: Visualize a frequência de aparição dos números nas predições
- **Análise de Tendências**: Identifique padrões e tendências nos resultados

### 4. Configurações
- **Seleção de Modelo**: Escolha qual modelo usar para as predições
- **Parâmetros de Predição**: Configure parâmetros específicos para cada modelo
- **Preferências de Interface**: Personalize a aparência e comportamento da interface

## Como Usar

### Iniciando o Sistema

1. **Inicie a API**:
   ```bash
   python api/iniciar_api.py
   ```
   A API estará disponível em `http://localhost:5000`

2. **Inicie o Servidor Web**:
   ```bash
   python -m http.server 8000 --directory interface
   ```
   A interface estará disponível em `http://localhost:8000`

### Fazendo Predições

1. **Acesse a Interface**: Abra `http://localhost:8000` no seu navegador
2. **Selecione o Modelo**: Use o dropdown para escolher o modelo desejado
3. **Configure Parâmetros**: Ajuste os parâmetros conforme necessário
4. **Gere Predição**: Clique em "Gerar Predição" para obter os números sugeridos
5. **Visualize Resultados**: Os números preditos aparecerão com suas respectivas probabilidades

### Monitorando Performance

1. **Acesse Estatísticas**: Clique na aba "Estatísticas" para ver métricas detalhadas
2. **Visualize Gráficos**: Use os gráficos interativos para analisar tendências
3. **Compare Modelos**: Compare a performance entre diferentes modelos
4. **Exporte Dados**: Baixe relatórios em formato JSON ou CSV

## Estrutura da Interface

```
interface/
├── index.html          # Página principal
├── css/
│   ├── style.css       # Estilos principais
│   └── dashboard.css   # Estilos do dashboard
├── js/
│   ├── app.js          # Lógica principal da aplicação
│   ├── api.js          # Comunicação com a API
│   ├── charts.js       # Gráficos e visualizações
│   └── utils.js        # Funções utilitárias
└── README.md           # Esta documentação
```

## APIs Integradas

### Endpoints Principais

- **POST /predict**: Gera predições usando o modelo especificado
- **GET /stats**: Retorna estatísticas de uso e performance
- **GET /models**: Lista todos os modelos disponíveis e seus status
- **GET /health**: Verifica o status de saúde da API

### Exemplo de Uso da API

```javascript
// Fazer uma predição
const response = await fetch('http://localhost:5000/predict', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        modelo: 'tensorflow_basico',
        numeros_entrada: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    })
});

const resultado = await response.json();
console.log('Números preditos:', resultado.numeros_preditos);
```

## Recursos Avançados

### Modo Escuro/Claro
- Alterne entre temas claro e escuro usando o botão no canto superior direito
- A preferência é salva automaticamente no localStorage

### Responsividade
- Interface totalmente responsiva, otimizada para desktop, tablet e mobile
- Layout adaptativo que se ajusta automaticamente ao tamanho da tela

### Notificações em Tempo Real
- Sistema de notificações para alertas importantes
- Atualizações automáticas de status dos modelos
- Notificações de sucesso/erro para ações do usuário

## Solução de Problemas

### Problemas Comuns

1. **API não responde**:
   - Verifique se a API está rodando em `http://localhost:5000`
   - Confirme que não há conflitos de porta
   - Verifique os logs da API para erros

2. **Interface não carrega**:
   - Verifique se o servidor web está rodando na porta 8000
   - Confirme que todos os arquivos estão no diretório `interface/`
   - Verifique o console do navegador para erros JavaScript

3. **Predições não funcionam**:
   - Verifique se os modelos estão carregados corretamente
   - Confirme que os parâmetros de entrada estão corretos
   - Verifique a conectividade com a API

### Logs e Debugging

- Use o console do navegador (F12) para ver logs detalhados
- Verifique a aba Network para problemas de comunicação com a API
- Os logs da API estão disponíveis no terminal onde ela foi iniciada

## Contribuição

Para contribuir com melhorias na interface:

1. Mantenha o código JavaScript limpo e bem documentado
2. Use CSS moderno com flexbox/grid para layouts
3. Teste a responsividade em diferentes dispositivos
4. Mantenha a compatibilidade com navegadores modernos
5. Documente novas funcionalidades neste README

## Suporte

Para suporte técnico ou dúvidas sobre o uso da interface, consulte:
- Logs da aplicação no console do navegador
- Documentação da API em `api/README.md`
- Código fonte comentado nos arquivos JavaScript