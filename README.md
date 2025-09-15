# Endurance OUT/Treino (.xlsx) — Streamlit App

App para análise de estatísticas de Endurance com **OUT** (novo OUT quando `Lap Tm > 120s`) e filtro por **Treino** (texto na **coluna I** da planilha). Compatível **somente com `.xlsx`** (usa `openpyxl`).

## Como usar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```
Abra a URL indicada (geralmente `http://localhost:8501`).

## Entrada (.xlsx)
- Cada **aba** representa uma sessão.
- Deve haver uma coluna de **tempo de volta** detectável automaticamente (ex.: `Lap Tm`, `Lap Time` etc.). 
- A **coluna I** deve conter o **nome/texto do Treino** (será lida como coluna `Treino`).  
- Se existir `SSTRAP Tm`, o app cria `SSTRAP` numérico automaticamente.

## O que o app faz
- Converte tempos para segundos (aceita `M:SS.mmm`, `mm:ss.s`, `s`).
- Cria `Out` e inicia **novo OUT quando `Lap Tm > 120s`**.
- Filtros por **sessão**, **Treino(s)** e **OUT**.
- Gráficos: **Boxplot**, **Linha** e **Dispersão**.
- Estatísticas descritivas.
- Exporta CSVs filtrados por sessão.

## Observações
- Se sua última aba for um resumo/extra, há um checkbox para **remover a última aba** antes da análise.
- O app **não** suporta `.xls`.
- Requisitos gerados em 2025-09-15.
