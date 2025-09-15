
import math
from typing import Optional, Iterable, Union, List
import re

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

st.set_page_config(page_title="📊 Análise Estatística Endurance (Outs)", layout="wide")

# ==================== Helpers de legenda ====================
def add_session_legend(ax, handles, position="right", title="Sessões", fontsize="x-small"):
    """Coloca a legenda em: 'right' (fora), 'top' (acima, em linha) ou 'hide'."""
    if position == "right":
        return ax.legend(
            handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0, fontsize=fontsize, title=title, frameon=True
        )
    elif position == "top":
        ncol = max(1, min(len(handles), 4))
        return ax.legend(
            handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.02, 1.0, 0.2),
            mode="expand", borderaxespad=0.0, ncol=ncol, fontsize=fontsize,
            title=title, frameon=True
        )
    elif position == "hide":
        return None
    else:
        return ax.legend(handles=handles, loc="upper right", fontsize=fontsize, title=title)

def apply_layout_for_legend(fig, position: str):
    """Reserva margem para a legenda conforme posição."""
    if position == "right":
        fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    elif position == "top":
        fig.tight_layout(rect=[0.0, 0.12, 1.0, 0.88])
    else:
        fig.tight_layout()

# ==================== Utils ====================
def time_to_seconds(t):
    """Converte 'M:SS.mmm' ou string numérica para segundos; retorna NaN se não parsear."""
    try:
        ts = str(t).replace(',', '.').strip()
        if ts == "" or ts.lower() in {"nan", "none"}:
            return pd.NA
        if ':' in ts:
            m, s = ts.split(':', 1)
            return int(m) * 60 + float(s)
        return float(ts)
    except:
        return pd.NA

def coerce_numeric(series: pd.Series) -> pd.Series:
    """Converte strings com ,/. e milhar em float de modo tolerante."""
    def smart_to_float(x):
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return pd.NA
        s = re.sub(r'\.(?=\d{3}\b)', '', s)
        s = s.replace(',', '.')
        try:
            return float(s)
        except:
            return pd.NA
    return series.apply(smart_to_float)

def find_lap_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if c.strip().lower() in {"lap", "leadlap", "lap #", "#lap"}]
    if candidates:
        if "Lap" not in df.columns:
            df.rename(columns={candidates[0]: "Lap"}, inplace=True)
        return "Lap"
    if "Lap" not in df.columns:
        df["Lap"] = range(1, len(df) + 1)
    return "Lap"

def find_lap_time_column(df: pd.DataFrame) -> Optional[str]:
    exacts = ["Lap Tm","LapTm","Lap Time","LapTime","Lap_Time","Best Lap Tm","Best Lap",
              "Lap Time (s)","LapTime(s)","LAP_TM","LAP_TIME"]
    for c in exacts:
        if c in df.columns:
            return c
    for c in df.columns:
        cl = c.lower().replace(" ", "").replace("_","")
        if "lap" in cl and ("tm" in cl or "time" in cl):
            return c
    best, score = None, 0.0
    for c in df.columns:
        conv = df[c].apply(time_to_seconds)
        ratio = conv.notna().mean()
        name_hit = (c.lower().endswith("tm") or "time" in c.lower() or "lap" in c.lower())
        if ratio >= 0.7 and name_hit and ratio > score:
            best, score = c, ratio
    return best

@st.cache_data
def load_excel(file) -> dict:
    return pd.read_excel(file, sheet_name=None)

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "SSTRAP Tm" in out.columns and "SSTRAP" not in out.columns:
        out["SSTRAP"] = coerce_numeric(out["SSTRAP Tm"])
    find_lap_column(out)
    for c in out.columns:
        if c == "SSTRAP Tm":
            continue
        if c.lower().endswith("tm"):
            out[c] = out[c].apply(time_to_seconds)
        else:
            maybe = coerce_numeric(out[c])
            if maybe.notna().mean() >= 0.5:
                out[c] = maybe
    return out

def derive_outs(df: pd.DataFrame, lap_time_col: str = "Lap Tm", threshold: float = 120.0) -> pd.DataFrame:
    """Cria a coluna 'Out' (1,2,3,...) e muda o Out quando Lap Tm > threshold (padrão 120s)."""
    if "Out" in df.columns:
        return df
    out_num = 1
    outs = []
    for t in df[lap_time_col].fillna(threshold + 1):
        outs.append(out_num)
        try:
            val = float(t)
        except:
            val = threshold + 1
        if val > threshold:
            out_num += 1
    new_df = df.copy()
    new_df["Out"] = outs
    return new_df

def get_filtered(df: pd.DataFrame,
                 out_choice: Union[str, int, Iterable[int]],
                 min_lap_seconds: float, max_lap_seconds: float,
                 is_time_metric: bool, treinos: Optional[List[str]] = None) -> pd.DataFrame:
    """Filtra por treino(s), OUT(s) e faixa de 'Lap Tm'."""
    d = df.copy()

    # Treinos
    if treinos and "Treino" in d.columns:
        tr_norm = [str(t) for t in treinos]
        d = d[d["Treino"].astype(str).isin(tr_norm)]

    # OUT(s)
    if "Out" in d.columns:
        if isinstance(out_choice, str) and out_choice == "All":
            pass
        else:
            if isinstance(out_choice, Iterable) and not isinstance(out_choice, (str, bytes)):
                outs_set = set(int(x) for x in out_choice)
                d = d[d["Out"].isin(outs_set)]
            else:
                d = d[d["Out"] == int(out_choice)]

    # Lap Tm
    if is_time_metric and "Lap Tm" in d.columns:
        d = d[pd.to_numeric(d["Lap Tm"], errors="coerce").notna()]
        d = d[d["Lap Tm"].between(float(min_lap_seconds), float(max_lap_seconds), inclusive="both")]

    return d

def parse_float_any(s: str) -> Optional[float]:
    if s is None: return None
    txt = str(s).strip().replace(" ", "")
    if txt == "": return None
    if "," in txt and "." in txt:
        last_comma, last_dot = txt.rfind(","), txt.rfind(".")
        if last_comma > last_dot:
            txt = txt.replace(".", "").replace(",", ".")
        else:
            txt = txt.replace(",", "")
    else:
        if "," in txt: txt = txt.replace(",", ".")
    try:
        return float(txt)
    except:
        return None

def float_input(label: str, default: float, min_value: float = 0.0, max_value: float = 1e9, key: str = None) -> float:
    raw = st.text_input(label, value=str(default).replace(".", ","), key=key)
    val = parse_float_any(raw)
    if val is None:
        st.caption("↳ Valor vazio/ inválido: usando padrão.")
        val = default
    val = max(min_value, min(max_value, val))
    return float(val)

def annotate_box(ax, bp, ys_list, idx, color, fs, dy):
    """Anota mediana, Q1/Q3 + Máximo/Mínimo em cada box."""
    data_i = np.array(ys_list[idx], dtype=float)
    q1 = float(np.percentile(data_i, 25))
    q3 = float(np.percentile(data_i, 75))
    med = float(np.median(data_i))
    y_min = float(np.min(data_i))
    y_max = float(np.max(data_i))

    median_line = bp["medians"][idx]
    median_line.set_color("black"); median_line.set_linewidth(2.0)
    x_mid = float(np.mean(median_line.get_xdata()))
    y_med = float(np.mean(median_line.get_ydata()))

    ax.text(x_mid, y_med, f"{med:.3f}", fontsize=fs, va="center", ha="center",
            color="black", bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.5, linewidth=0),
            clip_on=True, zorder=5)
    ax.text(x_mid, q3 + dy, f"{q3:.3f}", fontsize=fs, va="bottom", ha="center",
            color="black", bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.5, linewidth=0),
            clip_on=True, zorder=5)
    ax.text(x_mid, q1 - dy, f"{q1:.3f}", fontsize=fs, va="top", ha="center",
            color="black", bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.5, linewidth=0),
            clip_on=True, zorder=5)
    ax.text(x_mid, y_max + 2*dy, f"{y_max:.3f}", fontsize=fs, va="bottom", ha="center",
            color="black", bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.5, linewidth=0),
            clip_on=True, zorder=5)
    ax.text(x_mid, y_min - 2*dy, f"{y_min:.3f}", fontsize=fs, va="top", ha="center",
            color="black", bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.5, linewidth=0),
            clip_on=True, zorder=5)

# ==================== App ====================
def main():
    st.title("📊 Análise Estatística Endurance — usando Outs (>120s)")

    uploaded = st.file_uploader("Faça upload do arquivo Excel", type=["xlsx", "xls"])
    if not uploaded:
        st.info("Envie seu arquivo de estatísticas Endurance.")
        return

    sheets = load_excel(uploaded)

    # (Opcional) remover a última aba
    remove_last_sheet = st.checkbox("Remover última aba da planilha", value=True)
    if remove_last_sheet and len(sheets) >= 1:
        last_key = list(sheets.keys())[-1]
        del sheets[last_key]

    # Pré-processamento
    sheets_missing_laptm = []
    for name in list(sheets):
        df = preprocess_df(sheets[name])
        ltcol = find_lap_time_column(df)
        if ltcol:
            if ltcol != "Lap Tm":
                df.rename(columns={ltcol: "Lap Tm"}, inplace=True)
            df["Lap Tm"] = df["Lap Tm"].apply(time_to_seconds)
            # Derivar OUTs com threshold=120s
            df = derive_outs(df, lap_time_col="Lap Tm", threshold=120.0)
        else:
            df["Lap Tm"] = pd.NA
            df["Out"] = 1
            sheets_missing_laptm.append(name)
        sheets[name] = df

    if sheets_missing_laptm:
        with st.expander("Abas sem coluna de tempo de volta identificável"):
            st.write(", ".join(sheets_missing_laptm))

    # ---- Seleções principais
    default_p1 = [s for s in sheets if s.strip().endswith("P1")]
    sessions = st.multiselect(
        "Selecione sessões para análise",
        options=list(sheets.keys()),
        default=default_p1[:3] if default_p1 else list(sheets.keys())[:3]
    )
    if not sessions:
        st.warning("Selecione ao menos uma aba (ex.: ‘- P1’).")
        return

    # Seletor de Treino POR sessão (coluna 'Treino' = coluna I)
    session_treinos = {}
    st.markdown("**Selecione o(s) Treino(s) (coluna 'I' / 'Treino') por sessão:**")
    for s in sessions:
        df_s = sheets[s]
        if "Treino" in df_s.columns:
            opts = sorted(pd.Series(df_s["Treino"].dropna().astype(str).unique()))
        else:
            opts = []
        session_treinos[s] = st.multiselect(
            f"Treino(s) — {s}",
            options=opts,
            default=opts,
            key=f"treinos_{s}"
        )

    # Seletor de OUT(s) por sessão (multiselect atrelado ao gráfico principal e boxplot)
    session_outs = {}
    for s in sessions:
        if "Out" in sheets[s].columns:
            outs = sorted(pd.Series(sheets[s]["Out"]).dropna().astype(int).unique().tolist())
        else:
            outs = []
        session_outs[s] = st.multiselect(
            f"OUT(s) — {s}",
            options=outs,
            default=outs,  # por padrão, todos
            key=f"outs_{s}"
        )

    chart_type = st.selectbox("Tipo de gráfico", ["Boxplot", "Linha", "Dispersão"])
    x_axis_mode = st.selectbox("Eixo X", ["Amostragem", "Lap"])

    first_df = sheets[sessions[0]]
    time_cols = [c for c in first_df.columns if c.lower().endswith("tm") and c != "SSTRAP Tm"]
    metric_opts = list(time_cols)
    if "SSTRAP" in first_df.columns:
        metric_opts += ["SSTRAP"]
    if not metric_opts:
        st.error("Não encontrei colunas de tempo (*Tm) nem 'SSTRAP'.")
        return

    labels_map = {c: c for c in metric_opts}
    if "SSTRAP" in labels_map:
        labels_map["SSTRAP"] = "Velocidade Máxima (SSTRAP)"

    metric = st.selectbox("Selecione métrica", options=metric_opts, format_func=lambda x: labels_map[x])
    ylabel = labels_map[metric]
    is_time_metric = metric.lower().endswith("tm")

    # Filtros por tempo
    min_lap = float_input("Excluir voltas com 'Lap Tm' abaixo de (s) (valor mínimo)", default=0.0, key="min_lap_main")
    max_lap = float_input("Excluir voltas com 'Lap Tm' acima de (s)", default=60.0, key="max_lap_main")
    if max_lap < min_lap:
        st.warning("O máximo não pode ser menor que o mínimo. Ajustei o máximo para o mínimo.")
        max_lap = float(min_lap)

    # ---------------- Gráfico principal (Linha / Dispersão / Boxplot) ----------------
    # Monta séries respeitando sessões + treinos por sessão + OUT(s) por sessão
    series_x, series_y, labels = [], [], []
    filtered_exports = {}

    for s in sessions:
        outs_chosen = session_outs.get(s) or "All"
        df_f = get_filtered(sheets[s], outs_chosen, min_lap, max_lap, is_time_metric, treinos=session_treinos.get(s))

        if metric not in df_f.columns or df_f.empty:
            continue

        if isinstance(outs_chosen, list) and len(outs_chosen) > 1:
            # Uma série por OUT selecionado
            for out_n in outs_chosen:
                df_o = df_f[df_f["Out"] == out_n]
                if df_o.empty: 
                    continue
                lap_col = find_lap_column(df_o)
                if x_axis_mode == "Lap":
                    df_plot = df_o.sort_values(lap_col)
                    x = df_plot[lap_col].tolist()
                    y = pd.to_numeric(df_plot[metric], errors="coerce").tolist()
                else:
                    df_plot = df_o.sort_values(metric)
                    x = list(range(1, len(df_plot) + 1))
                    y = pd.to_numeric(df_plot[metric], errors="coerce").tolist()
                series_x.append(x); series_y.append(y)
                labels.append(f"{s} — Out {int(out_n)}")
        else:
            # Uma série agregada (All ou único OUT)
            lap_col = find_lap_column(df_f)
            if x_axis_mode == "Lap":
                df_plot = df_f.sort_values(lap_col)
                x = df_plot[lap_col].tolist()
                y = pd.to_numeric(df_plot[metric], errors="coerce").tolist()
            else:
                df_plot = df_f.sort_values(metric)
                x = list(range(1, len(df_plot) + 1))
                y = pd.to_numeric(df_plot[metric], errors="coerce").tolist()
            series_x.append(x); series_y.append(y)
            if outs_chosen == "All":
                labels.append(f"{s} — Out(s): Todos")
            else:
                labels.append(f"{s} — Out {int(outs_chosen[0])}" if isinstance(outs_chosen, list) else f"{s} — Out {int(outs_chosen)}")

        # Export por sessão já filtrada
        filtered_exports[s] = df_f.copy()

    # Render principal
    fig, ax = plt.subplots(figsize=(10, 4))
    if chart_type == "Boxplot":
        valid_pairs = [(ys, lbl) for ys, lbl in zip(series_y, labels) if len(ys) > 0]
        if not valid_pairs:
            st.warning("Sem dados para plotar.")
            return
        ys_list, lbls = zip(*valid_pairs)
        bp = ax.boxplot(ys_list, patch_artist=True)
        cycle = plt.rcParams.get("axes.prop_cycle", None)
        base_colors = cycle.by_key().get("color", ["C0"]) if cycle else ["C0"]
        n_boxes = len(lbls); h_in = fig.get_size_inches()[1]
        fs = max(6, min(12, (10 * (h_in / 4.0)) * (8 / max(6, n_boxes))))
        all_vals = np.concatenate([np.array(v, dtype=float) for v in ys_list])
        y_range = float(np.nanmax(all_vals) - np.nanmin(all_vals)) if all_vals.size else 1.0
        dy = max(0.002 * y_range, 0.0005)
        for i, box in enumerate(bp["boxes"]):
            col = base_colors[i % len(base_colors)]
            box.set_facecolor(col); box.set_edgecolor(col)
            bp["whiskers"][2*i].set_color(col); bp["whiskers"][2*i + 1].set_color(col)
            bp["caps"][2*i].set_color(col); bp["caps"][2*i + 1].set_color(col)
            annotate_box(ax, bp, ys_list, i, col, fs, dy)
        handles_main = [mpatches.Patch(facecolor=base_colors[i % len(base_colors)],
                                       edgecolor=base_colors[i % len(base_colors)], label=l)
                        for i, l in enumerate(lbls)]
        ax.legend(handles=handles_main, loc="upper right", fontsize="xx-small")
        ax.set_xticks([])
    elif chart_type == "Linha":
        for x, y, lbl in zip(series_x, series_y, labels):
            if len(x) and len(y):
                ax.plot(x, y, label=lbl)
        ax.legend(loc="upper right", fontsize="xx-small")
        ax.set_xlabel("Lap" if x_axis_mode == "Lap" else "Amostra")
        ax.set_ylabel(ylabel)
    else:  # Dispersão
        for x, y, lbl in zip(series_x, series_y, labels):
            if len(x) and len(y):
                ax.scatter(x, y, s=10, label=lbl)
        ax.legend(loc="upper right", fontsize="xx-small")
        ax.set_xlabel("Lap" if x_axis_mode == "Lap" else "Amostra")
        ax.set_ylabel(ylabel)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Estatísticas descritivas por amostragem
    st.header("📊 Estatísticas Descritivas por Amostragem")
    stats_frames = []
    for lbl, y in zip(labels, series_y):
        y_clean = pd.Series(pd.to_numeric(y, errors="coerce")).dropna()
        if not y_clean.empty:
            stats_frames.append(y_clean.describe().to_frame(name=lbl))
    if stats_frames:
        st.dataframe(pd.concat(stats_frames, axis=1))
    else:
        st.info("Sem dados suficientes para estatísticas descritivas.")

    # Downloads por sessão
    st.subheader("⬇️ Baixar dados filtrados (por sessão)")
    for s in sessions:
        if s in filtered_exports and not filtered_exports[s].empty:
            st.download_button(
                label=f"Baixar '{s}' (CSV)",
                data=filtered_exports[s].to_csv(index=False).encode("utf-8"),
                file_name=f"{s}_filtrado.csv",
                mime="text/csv",
                key=f"dl_{s}"
            )

if __name__ == "__main__":
    main()
