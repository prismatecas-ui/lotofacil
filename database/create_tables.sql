-- =====================================================
-- SCRIPT DE CRIAÇÃO DAS TABELAS SQLITE - LOTOFÁCIL
-- =====================================================
-- Criado em: Janeiro 2025
-- Baseado na análise dos arquivos:
--   - base/base_dados.xlsx (1.994 linhas × 58 colunas)
--   - base/resultados.csv (1.991 linhas × 18 colunas)
-- =====================================================

-- Habilitar chaves estrangeiras
PRAGMA foreign_keys = ON;

-- =====================================================
-- TABELA PRINCIPAL: SORTEIOS
-- Armazena informações básicas de cada concurso
-- =====================================================
CREATE TABLE IF NOT EXISTS sorteios (
    concurso INTEGER PRIMARY KEY,
    data_sorteio DATE NOT NULL,
    dia_semana TEXT,
    mes INTEGER CHECK (mes BETWEEN 1 AND 12),
    ano INTEGER CHECK (ano >= 2003),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- TABELA: NUMEROS_SORTEADOS
-- Armazena os 15 números de cada sorteio
-- =====================================================
CREATE TABLE IF NOT EXISTS numeros_sorteados (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    concurso INTEGER NOT NULL,
    posicao INTEGER NOT NULL CHECK (posicao BETWEEN 1 AND 15),
    numero INTEGER NOT NULL CHECK (numero BETWEEN 1 AND 25),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (concurso) REFERENCES sorteios(concurso) ON DELETE CASCADE,
    UNIQUE(concurso, posicao)
);

-- =====================================================
-- TABELA: RESULTADOS_JOGOS
-- Armazena resultados de jogos/apostas
-- =====================================================
CREATE TABLE IF NOT EXISTS resultados_jogos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    concurso INTEGER NOT NULL,
    acertos INTEGER NOT NULL CHECK (acertos >= 0 AND acertos <= 15),
    premio DECIMAL(15,2) DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (concurso) REFERENCES sorteios(concurso) ON DELETE CASCADE
);

-- =====================================================
-- TABELA: ESTATISTICAS_NUMEROS
-- Armazena estatísticas de frequência dos números
-- =====================================================
CREATE TABLE IF NOT EXISTS estatisticas_numeros (
    numero INTEGER PRIMARY KEY CHECK (numero BETWEEN 1 AND 25),
    frequencia_total INTEGER DEFAULT 0,
    frequencia_recente INTEGER DEFAULT 0, -- últimos 100 sorteios
    ultima_aparicao INTEGER, -- último concurso que apareceu
    maior_atraso INTEGER DEFAULT 0,
    atraso_atual INTEGER DEFAULT 0,
    media_intervalo DECIMAL(10,2) DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- TABELA: PADROES_SORTEIOS
-- Armazena padrões identificados nos sorteios
-- =====================================================
CREATE TABLE IF NOT EXISTS padroes_sorteios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    concurso INTEGER NOT NULL,
    numeros_pares INTEGER DEFAULT 0,
    numeros_impares INTEGER DEFAULT 0,
    soma_numeros INTEGER DEFAULT 0,
    numero_menor INTEGER,
    numero_maior INTEGER,
    amplitude INTEGER, -- diferença entre maior e menor
    sequencias INTEGER DEFAULT 0, -- quantidade de números em sequência
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (concurso) REFERENCES sorteios(concurso) ON DELETE CASCADE
);

-- =====================================================
-- TABELA: CONFIGURACOES
-- Armazena configurações do sistema
-- =====================================================
CREATE TABLE IF NOT EXISTS configuracoes (
    chave TEXT PRIMARY KEY,
    valor TEXT NOT NULL,
    descricao TEXT,
    tipo TEXT DEFAULT 'string', -- string, integer, boolean, decimal
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- TABELA: LOGS_SISTEMA
-- Armazena logs de operações do sistema
-- =====================================================
CREATE TABLE IF NOT EXISTS logs_sistema (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nivel TEXT NOT NULL CHECK (nivel IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    modulo TEXT NOT NULL,
    mensagem TEXT NOT NULL,
    detalhes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- ÍNDICES PARA PERFORMANCE
-- =====================================================

-- Índices para tabela sorteios
CREATE INDEX IF NOT EXISTS idx_sorteios_data ON sorteios(data_sorteio);
CREATE INDEX IF NOT EXISTS idx_sorteios_ano_mes ON sorteios(ano, mes);

-- Índices para tabela numeros_sorteados
CREATE INDEX IF NOT EXISTS idx_numeros_concurso ON numeros_sorteados(concurso);
CREATE INDEX IF NOT EXISTS idx_numeros_numero ON numeros_sorteados(numero);
CREATE INDEX IF NOT EXISTS idx_numeros_posicao ON numeros_sorteados(posicao);

-- Índices para tabela resultados_jogos
CREATE INDEX IF NOT EXISTS idx_resultados_concurso ON resultados_jogos(concurso);
CREATE INDEX IF NOT EXISTS idx_resultados_acertos ON resultados_jogos(acertos);

-- Índices para tabela padroes_sorteios
CREATE INDEX IF NOT EXISTS idx_padroes_concurso ON padroes_sorteios(concurso);
CREATE INDEX IF NOT EXISTS idx_padroes_soma ON padroes_sorteios(soma_numeros);

-- Índices para tabela logs_sistema
CREATE INDEX IF NOT EXISTS idx_logs_nivel ON logs_sistema(nivel);
CREATE INDEX IF NOT EXISTS idx_logs_modulo ON logs_sistema(modulo);
CREATE INDEX IF NOT EXISTS idx_logs_data ON logs_sistema(created_at);

-- =====================================================
-- TRIGGERS PARA ATUALIZAÇÃO AUTOMÁTICA
-- =====================================================

-- Trigger para atualizar updated_at em sorteios
CREATE TRIGGER IF NOT EXISTS trigger_sorteios_updated_at
    AFTER UPDATE ON sorteios
    FOR EACH ROW
BEGIN
    UPDATE sorteios SET updated_at = CURRENT_TIMESTAMP WHERE concurso = NEW.concurso;
END;

-- Trigger para atualizar updated_at em estatisticas_numeros
CREATE TRIGGER IF NOT EXISTS trigger_estatisticas_updated_at
    AFTER UPDATE ON estatisticas_numeros
    FOR EACH ROW
BEGIN
    UPDATE estatisticas_numeros SET updated_at = CURRENT_TIMESTAMP WHERE numero = NEW.numero;
END;

-- Trigger para atualizar updated_at em configuracoes
CREATE TRIGGER IF NOT EXISTS trigger_configuracoes_updated_at
    AFTER UPDATE ON configuracoes
    FOR EACH ROW
BEGIN
    UPDATE configuracoes SET updated_at = CURRENT_TIMESTAMP WHERE chave = NEW.chave;
END;

-- =====================================================
-- VIEWS PARA CONSULTAS FREQUENTES
-- =====================================================

-- View: Sorteios com números em uma linha
CREATE VIEW IF NOT EXISTS vw_sorteios_completos AS
SELECT 
    s.concurso,
    s.data_sorteio,
    s.dia_semana,
    GROUP_CONCAT(n.numero ORDER BY n.posicao) as numeros_sorteados,
    p.numeros_pares,
    p.numeros_impares,
    p.soma_numeros,
    p.amplitude
FROM sorteios s
LEFT JOIN numeros_sorteados n ON s.concurso = n.concurso
LEFT JOIN padroes_sorteios p ON s.concurso = p.concurso
GROUP BY s.concurso, s.data_sorteio, s.dia_semana, p.numeros_pares, p.numeros_impares, p.soma_numeros, p.amplitude
ORDER BY s.concurso;

-- View: Estatísticas resumidas por número
CREATE VIEW IF NOT EXISTS vw_estatisticas_resumo AS
SELECT 
    numero,
    frequencia_total,
    frequencia_recente,
    ultima_aparicao,
    atraso_atual,
    ROUND(frequencia_total * 100.0 / (SELECT COUNT(*) FROM sorteios), 2) as percentual_aparicao
FROM estatisticas_numeros
ORDER BY frequencia_total DESC;

-- =====================================================
-- INSERÇÃO DE DADOS INICIAIS
-- =====================================================

-- Configurações iniciais do sistema
INSERT OR IGNORE INTO configuracoes (chave, valor, descricao, tipo) VALUES
('versao_banco', '1.0.0', 'Versão do esquema do banco de dados', 'string'),
('ultimo_concurso_processado', '0', 'Último concurso processado na migração', 'integer'),
('data_ultima_atualizacao', '', 'Data da última atualização dos dados', 'string'),
('modo_debug', 'false', 'Ativar modo debug do sistema', 'boolean'),
('backup_automatico', 'true', 'Ativar backup automático', 'boolean'),
('intervalo_backup_horas', '24', 'Intervalo entre backups em horas', 'integer'),
('api_caixa_ativa', 'false', 'API da Caixa está ativa', 'boolean'),
('modelo_treinado', 'false', 'Modelo de ML está treinado', 'boolean');

-- Inicializar estatísticas para todos os números (1-25)
INSERT OR IGNORE INTO estatisticas_numeros (numero) 
SELECT value FROM (
    WITH RECURSIVE numeros(value) AS (
        SELECT 1
        UNION ALL
        SELECT value + 1 FROM numeros WHERE value < 25
    )
    SELECT value FROM numeros
);

-- =====================================================
-- COMENTÁRIOS E DOCUMENTAÇÃO
-- =====================================================

/*
ESTRUTURA DO BANCO DE DADOS - RESUMO:

1. SORTEIOS: Tabela principal com dados básicos dos concursos
2. NUMEROS_SORTEADOS: Números de cada sorteio (normalizada)
3. RESULTADOS_JOGOS: Resultados de apostas/jogos
4. ESTATISTICAS_NUMEROS: Frequências e estatísticas dos números
5. PADROES_SORTEIOS: Padrões matemáticos identificados
6. CONFIGURACOES: Configurações do sistema
7. LOGS_SISTEMA: Logs de operações

RELACIONAMENTOS:
- sorteios (1) -> numeros_sorteados (15)
- sorteios (1) -> resultados_jogos (N)
- sorteios (1) -> padroes_sorteios (1)

ÍNDICES CRIADOS PARA:
- Consultas por data
- Consultas por número
- Consultas por concurso
- Consultas por padrões
- Logs do sistema

TRIGGERS:
- Atualização automática de timestamps
- Manutenção de integridade referencial

VIEWS:
- Consultas otimizadas para relatórios
- Estatísticas resumidas
*/

-- =====================================================
-- FIM DO SCRIPT
-- =====================================================