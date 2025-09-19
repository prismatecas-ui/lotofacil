# Guia do Usuário - Sistema de Jogos Recomendados Lotofácil

## Como Usar o Sistema

### 1. Iniciando o Sistema

**Passo 1**: Abra o terminal e navegue até a pasta do projeto

**Passo 2**: Inicie a API do sistema:
```bash
python api/iniciar_api.py
```

**Passo 3**: Em outro terminal, inicie a interface web:
```bash
python -m http.server 8000 --directory interface
```

**Passo 4**: Abra seu navegador e acesse: `http://localhost:8000`

### 1.1. Monitoramento do Sistema

Para monitorar o sistema em tempo real, use o script de monitoramento:
```bash
python experimentos/monitor_simples.py
```

Este script mostra:
- Processos Python ativos
- Arquivos recentes modificados
- Progresso de otimizações
- Tempo de execução
- Status do sistema

### 2. Gerando Jogos Recomendados

1. **Acesse a seção "Jogos Recomendados"** no menu principal

2. **Escolha o Modelo**: No dropdown "Modelo", selecione:
   - **TensorFlow Básico**: Modelo básico de rede neural
   - **Algoritmos Avançados**: Para análises mais complexas
   - **Análise de Padrões**: Para identificar padrões específicos

3. **Configure os Números** (opcional):
   - Use a grade de números para selecionar manualmente (15 números)
   - Ou deixe em branco para recomendação automática
   - O contador mostra quantos números foram selecionados

4. **Opções Adicionais**:
   - ☑️ **Usar cache para melhor performance**: Melhora velocidade de resposta

5. **Clique em "Gerar Jogos"**

6. **Veja os Resultados**:
   - 15 números recomendados para o próximo sorteio
   - Nível de confiança da recomendação
   - Tempo de processamento

### 3. Entendendo os Resultados

#### Números Recomendados
- São os 15 números que o modelo considera mais prováveis
- Ordenados por probabilidade (do mais provável ao menos provável)

#### Confiança
- Percentual que indica a "certeza" do modelo
- Valores mais altos indicam maior confiança na recomendação
- **Modelo Atual**: 81.09% de acurácia geral

#### Probabilidades
- Mostra a probabilidade individual de cada número
- Valores entre 0 e 1 (quanto maior, melhor)

#### Performance dos Modelos Disponíveis
- **Modelo Final Extremo**: 81.09% acurácia, 66.60% precisão, 73.13% F1-score
- **Ensemble Otimizado**: Combina Random Forest, Gradient Boosting e Extra Trees
- **Técnicas Aplicadas**: Feature Engineering Extrema, SMOTETomek, Grid Search

### 4. Visualizando Estatísticas

1. **Clique na aba "Estatísticas"** para ver:
   - **Jogos Recomendados Hoje**: Quantidade de jogos gerados hoje
   - **Precisão Média**: Taxa média de acerto dos modelos
   - **Melhor Modelo**: Modelo com melhor performance atual
   - **Última Atualização**: Horário da última atualização dos dados

2. **Gráficos Interativos Disponíveis**:
   - **Análise de Performance**: Evolução da precisão ao longo do tempo
   - **Precisão por Modelo**: Comparação entre diferentes modelos
   - **Frequência dos Números**: Análise de frequência histórica
   - **Tendências Temporais**: Padrões ao longo do tempo

3. **Tabelas Detalhadas**:
   - **Performance por Modelo**: Precisão, jogos recomendados, acertos, tempo médio
   - **Histórico Recente**: Data/hora, modelo usado, números preditos, confiança, resultado

4. **Análise de Números**:
   - **Números Mais Frequentes**: Os que mais aparecem nos sorteios
   - **Números Menos Frequentes**: Os que menos aparecem
   - **Números em Tendência**: Padrões recentes identificados

### 5. Configurações

#### 5.1 Configurações Básicas

1. **Acesse as Configurações** clicando na aba "Configurações" no menu principal
2. **Configurações Gerais**:
   - **Tema**: Alterne entre modo claro e escuro
   - **Idioma**: Selecione o idioma da interface
   - **Auto-atualização**: Ative para atualizar dados automaticamente

#### 5.2 Configurações Avançadas - Parâmetros de Predição

**Limite de Confiança**
- Define o nível mínimo de confiança para aceitar uma recomendação
- Valores entre 0.1 e 1.0
- Padrão: 0.7 (70%)
- Valores mais altos = recomendações mais conservadoras

**Profundidade Histórica**
- Quantidade de sorteios anteriores considerados na análise
- Valores entre 10 e 500 sorteios
- Padrão: 100 sorteios
- Mais dados = análise mais robusta, mas processamento mais lento

**Quantidade de Predições**
- Número de jogos recomendados a serem gerados
- Valores entre 1 e 10
- Padrão: 3 jogos

#### 5.3 Parâmetros de Algoritmo

**Peso dos Padrões**
- Influência dos padrões históricos na recomendação
- Valores entre 0.0 e 1.0
- Padrão: 0.4
- Maior peso = mais importância aos padrões identificados

**Peso da Frequência**
- Influência da frequência histórica dos números
- Valores entre 0.0 e 1.0
- Padrão: 0.3
- Maior peso = favorece números mais frequentes

**Peso das Tendências**
- Influência das tendências recentes
- Valores entre 0.0 e 1.0
- Padrão: 0.3
- Maior peso = favorece números em tendência de alta

**Modo do Algoritmo**
- **Conservador**: Prioriza números com histórico consistente
- **Balanceado**: Equilibra todos os fatores (padrão)
- **Agressivo**: Favorece tendências recentes e padrões emergentes

**Intervalo de Atualização**
- Frequência de atualização automática dos dados
- Opções: 30s, 1min, 5min, 15min, 30min
- Padrão: 5 minutos

#### 5.4 Opções Avançadas

**☑️ Excluir Números Recentes**
- Remove números que saíram nos últimos sorteios
- Útil para evitar repetições imediatas

**☑️ Incluir Números Quentes**
- Prioriza números que estão "quentes" (saindo com frequência)
- Baseado em análise de tendências recentes

**☑️ Usar Machine Learning**
- Ativa algoritmos de aprendizado de máquina avançados
- Melhora precisão mas aumenta tempo de processamento

**☑️ Auto-atualização**
- Atualiza automaticamente os dados e modelos
- Recomendado para manter sistema sempre atualizado

**☑️ Habilitar WebSocket**
- Ativa conexão em tempo real para atualizações instantâneas
- Melhora experiência do usuário

#### 5.5 Botões de Ação

- **Resetar**: Volta todas as configurações para os valores padrão
- **Salvar**: Salva as configurações atuais
- **Aplicar**: Aplica as configurações sem salvar permanentemente

#### Tema da Interface
- **Modo Claro**: Interface com fundo branco (padrão)
- **Modo Escuro**: Interface com fundo escuro
- Clique no ícone de lua/sol no canto superior direito

#### Notificações
- O sistema mostra notificações para:
  - ✅ Previsões realizadas com sucesso
  - ⚠️ Avisos sobre parâmetros
  - ❌ Erros de conexão ou processamento

### 6. Notificações

#### 6.1 Configurações de Notificações

**Notificações do Navegador**
- ☑️ Ativar para receber notificações no navegador
- Requer permissão do navegador

**Notificações Sonoras**
- ☑️ Ativar para receber alertas sonoros
- Útil quando a aba não está em foco

#### 6.2 Tipos de Notificações

**☑️ Novas Recomendações**
- Alerta quando novos jogos são recomendados
- Inclui resumo dos números sugeridos

**☑️ Resultados de Jogos**
- Notifica sobre resultados de sorteios
- Compara com suas recomendações anteriores

**☑️ Alertas de Performance**
- Informa sobre mudanças na performance dos modelos
- Alerta sobre modelos com baixa precisão

**☑️ Atualizações do Sistema**
- Notifica sobre atualizações de modelos
- Informa sobre novas funcionalidades

#### 6.3 Gerenciamento de Notificações

**Filtros Disponíveis**:
- **Todas**: Mostra todas as notificações
- **Não Lidas**: Apenas notificações não visualizadas
- **Importantes**: Apenas notificações de alta prioridade
- **Sistema**: Apenas notificações do sistema

**Ações**:
- **Marcar como Lida**: Marca notificação como visualizada
- **Excluir**: Remove notificação da lista
- **Limpar Todas**: Remove todas as notificações

### 7. Dicas de Uso

#### Para Melhores Resultados:
1. **Use modelos com maior acurácia** para jogos recomendados mais precisos
2. **Combine diferentes modelos** para validar resultados
3. **Observe a confiança** - valores acima de 70% são mais confiáveis
4. **Analise as estatísticas** para entender padrões
5. **Configure os parâmetros avançados** conforme sua estratégia
6. **Mantenha o sistema atualizado** para melhor performance

#### Interpretando a Confiança:
- **Acima de 80%**: Alta confiança (excelente)
- **70-80%**: Boa confiança (recomendado)
- **60-70%**: Confiança moderada (use com cautela)
- **Abaixo de 50%**: Baixa confiança (considere outro modelo)

#### Estratégias de Configuração:
- **Conservadora**: Limite de confiança alto (0.8+), profundidade histórica alta (200+)
- **Balanceada**: Configurações padrão do sistema
- **Agressiva**: Peso das tendências alto (0.5+), incluir números quentes ativado

### 8. Problemas Comuns e Soluções

#### "Erro de Conexão com a API"
**Causa**: Servidor da API não está rodando
**Solução**: 
1. Abra o terminal na pasta do projeto
2. Execute: `python api/iniciar_api.py`
3. Aguarde a mensagem "API rodando em http://localhost:5000"

#### "Modelo não encontrado"
**Causa**: Arquivo do modelo foi movido ou deletado
**Solução**:
1. Verifique se os arquivos `.pkl` estão na pasta `modelos/`
2. Se não estiver, execute o treinamento: `python treinar_modelo.py`

#### "Interface não carrega"
**Causa**: Servidor web não está ativo
**Solução**:
1. Abra terminal na pasta `interface/`
2. Execute: `python -m http.server 8000`
3. Acesse: http://localhost:8000

#### "Números inválidos"
**Causa**: Formato incorreto na entrada manual
**Solução**:
- Use apenas números de 1 a 25
- Selecione exatamente 15 números na grade
- Para entrada manual: separe por vírgula, sem espaços
- Exemplo correto: `1,2,3,4,5,6,7,8,9,10,11,12,13,14,15`

#### "Configurações não salvam"
**Causa**: Problemas de permissão ou cache
**Solução**:
1. Limpe o cache do navegador
2. Verifique se JavaScript está habilitado
3. Tente usar o botão "Aplicar" antes de "Salvar"

#### "Notificações não funcionam"
**Causa**: Permissões do navegador não concedidas
**Solução**:
1. Clique no ícone de cadeado na barra de endereços
2. Permita notificações para o site
3. Verifique se as notificações estão ativadas nas configurações

### 9. Recursos Mobile

A interface funciona perfeitamente em dispositivos móveis:
- **Toque** nos botões em vez de clicar
- **Deslize** para navegar pelos gráficos
- **Gire a tela** para melhor visualização dos gráficos
- **Menu hambúrguer** aparece em telas pequenas

### 10. Exportação de Dados

#### 10.1 Formatos de Exportação Disponíveis

**Exportar como JSON**
- Formato estruturado para integração com outros sistemas
- Inclui metadados completos (timestamps, confiança, modelo usado)
- Ideal para desenvolvedores e análises automatizadas

**Exportar como CSV**
- Formato tabular compatível com Excel e planilhas
- Dados organizados em colunas (data, números, confiança, modelo)
- Ideal para análises estatísticas e relatórios

#### 10.2 Opções de Exportação

**Dados a Exportar**:
- **Jogos Recomendados**: Apenas os números recomendados
- **Histórico Completo**: Todas as recomendações anteriores
- **Estatísticas**: Dados de performance e análises
- **Configurações**: Backup das configurações atuais

**Período**:
- **Hoje**: Apenas dados do dia atual
- **Última Semana**: Dados dos últimos 7 dias
- **Último Mês**: Dados dos últimos 30 dias
- **Personalizado**: Selecione período específico

#### 10.3 Outras Opções

**Limpar Cache**
- Remove dados temporários armazenados
- Útil para resolver problemas de performance
- **Atenção**: Pode aumentar tempo de carregamento inicial

**Copiar Números**
- Copia números recomendados para área de transferência
- Formato: números separados por vírgula
- Útil para colar em outros aplicativos

### 11. Suporte

**Se precisar de ajuda**:
1. Verifique este guia primeiro
2. Consulte os logs no console do navegador (F12)
3. Verifique os logs da API no terminal
4. Reinicie o sistema completo se necessário

**Lembre-se**: Este é um sistema de previsão baseado em análise de padrões históricos. Os resultados são sugestões e não garantem acertos nos sorteios.

---

### 12. Endpoints da API

#### Principais Endpoints:
- `GET /health` - Status da API
- `POST /predict` - Fazer previsão
- `GET /models` - Listar modelos disponíveis
- `GET /metrics` - Métricas do sistema
- `GET /cache/status` - Status do cache

#### Exemplo de Uso da API:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"modelo": "modelo_final_extremo", "numeros": []}'
```

### 13. Arquivos Importantes

#### Modelos:
- `modelos/modelo_final_extremo_20250919_113536.pkl` - Modelo principal (81.09% acurácia)
- `modelos/ensemble_otimizado_20250919_103106.pkl` - Ensemble otimizado
- `modelos/modelo_super_otimizado_20250919_103719.pkl` - Versão anterior

#### Scripts de Monitoramento:
- `experimentos/monitor_simples.py` - Monitor em tempo real
- `experimentos/monitor_otimizacao.py` - Monitor de otimizações

#### Configurações:
- `api/config.py` - Configurações da API
- `api/config_api.json` - Configurações JSON

---

**Versão do Sistema**: 2.0  
**Última Atualização**: Setembro 2025  
**Modelo Atual**: Extremo v20250919_113536 (81.09% acurácia)  
**Compatibilidade**: Navegadores modernos (Chrome, Firefox, Safari, Edge)