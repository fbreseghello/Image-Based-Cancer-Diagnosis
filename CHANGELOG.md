# Changelog - Image-Based Cancer Diagnosis

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

---

## [2.0.0] - 2026-01-07

### 🎉 Atualização Completa do Projeto

### Adicionado

#### Arquitetura e Treinamento
- ✨ Arquitetura CNN aprimorada com BatchNormalization e Dropout
- ✨ Suporte para Transfer Learning com MobileNetV2
- ✨ Data augmentation nativo (rotação, zoom, flip, translação)
- ✨ Sistema de callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
- ✨ Integração com TensorBoard para monitoramento em tempo real
- ✨ Interface de linha de comando (argparse) para configurações
- ✨ Múltiplas métricas de avaliação (Precision, Recall, AUC)
- ✨ Visualização e salvamento do histórico de treinamento

#### Interface e Usabilidade
- ✨ Interface Streamlit completamente redesenhada
- ✨ Visualização Grad-CAM para interpretabilidade do modelo
- ✨ Métricas de confiança com níveis visuais
- ✨ Comparação entre predição e rótulo verdadeiro para amostras
- ✨ Informações detalhadas do modelo
- ✨ CSS customizado para melhor aparência
- ✨ Gráficos de treinamento aprimorados

#### Avaliação e Métricas
- ✨ Script dedicado `evaluate_model.py` para avaliação completa
- ✨ Matriz de confusão com visualização
- ✨ Curva ROC com cálculo de AUC
- ✨ Curva Precision-Recall
- ✨ Distribuição de probabilidades preditas
- ✨ Relatório de classificação detalhado
- ✨ Salvamento automático de todas as visualizações

#### Estrutura e Organização
- ✨ Arquivo `src/config.py` centralizando todas as configurações
- ✨ Logging estruturado com Python logging
- ✨ Tratamento de erros robusto com exceções customizadas
- ✨ Validação de entrada de dados
- ✨ Funções utilitárias aprimoradas em `model_utils.py`

#### Documentação
- ✨ README.md completo e profissional
- ✨ GUIA_USO.md com instruções detalhadas em português
- ✨ Docstrings em todas as funções
- ✨ Comentários explicativos no código
- ✨ Changelog para rastrear mudanças

#### DevOps e Utilidades
- ✨ Script `setup.py` para configuração automática
- ✨ Script `run_app.py` para execução simplificada
- ✨ `.gitignore` atualizado para projetos ML/Python
- ✨ `.gitkeep` para rastreamento de diretórios vazios
- ✨ `requirements.txt` com versões específicas

### Melhorado

#### Código
- 🔨 Modularização do código em funções reutilizáveis
- 🔨 Separação de concerns (config, utils, app, train)
- 🔨 Performance otimizada com cache e prefetch de dados
- 🔨 Compatibilidade com versões modernas do TensorFlow/Keras

#### Modelo
- 🔨 De CNN simples (3 camadas) para arquitetura profunda (9+ camadas)
- 🔨 Adição de regularização L2
- 🔨 Normalização de batch para estabilidade
- 🔨 Dropout adaptativo em múltiplas camadas

#### Interface
- 🔨 Layout responsivo com colunas
- 🔨 Feedback visual melhorado (cores, ícones)
- 🔨 Mensagens de erro mais informativas
- 🔨 Loading states e spinners

### Removido

- ❌ Código hardcoded e valores mágicos
- ❌ Imports desnecessários
- ❌ Configurações inline (movidas para config.py)

### Corrigido

- 🐛 Tratamento de exceções para carregamento de modelo
- 🐛 Validação de paths de arquivos
- 🐛 Compatibilidade com múltiplos formatos de imagem
- 🐛 Problemas de normalização de imagens

### Dependências

#### Atualizadas
- `tensorflow`: >= 2.15.0 (antes: sem versão específica)
- `streamlit`: >= 1.28.0 (antes: sem versão específica)
- `numpy`: >= 1.24.0 (antes: sem versão específica)
- `pillow`: >= 10.0.0 (antes: sem versão específica)

#### Adicionadas
- `keras`: >= 3.0.0
- `scikit-learn`: >= 1.3.0
- `opencv-python`: >= 4.8.0
- `matplotlib`: >= 3.7.0
- `seaborn`: >= 0.12.0
- `pandas`: >= 2.0.0
- `tqdm`: >= 4.65.0
- `pytest`: >= 7.4.0
- `black`: >= 23.0.0
- `flake8`: >= 6.0.0

---

## [1.0.0] - Data Original

### Inicial

- CNN básica com 2 camadas convolucionais
- Script de treinamento simples
- Interface Streamlit básica
- Upload e predição de imagens
- Visualização de amostras
- Gráfico de histórico de treinamento

---

## Planejamento Futuro

### [3.0.0] - Próximas Funcionalidades

#### Planejado
- [ ] Suporte a classificação multi-classe
- [ ] API REST para integração com outros sistemas
- [ ] Containerização com Docker
- [ ] CI/CD com GitHub Actions
- [ ] Testes unitários e de integração
- [ ] Suporte a DICOM (formato médico padrão)
- [ ] Ensemble de múltiplos modelos
- [ ] Interface mobile
- [ ] Autenticação e autorização
- [ ] Banco de dados para histórico de predições
- [ ] Dashboard de analytics
- [ ] Suporte a múltiplos idiomas

---

**Formato do Changelog baseado em [Keep a Changelog](https://keepachangelog.com/)**

**Versionamento baseado em [Semantic Versioning](https://semver.org/)**
