
import math
from typing import Optional, Iterable, Union, List, Dict
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
    if position == "right":
        return ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                         borderaxespad=0.0, fontsize=fontsize, title=title, frameon=True)
    elif position == "top":
        ncol = max(1, min(len(handles), 4))
        return ax.legend(handles=handles, loc="lower left",
                         bbox_to_anchor=(0.0, 1.02, 1.0, 0.2),
                         mode="expand", borderaxespad=0.0, ncol=ncol,
                         fontsize=fontsize, title=title, frameon=True)
    elif position == "hide":
        return None
    else:
        return ax.legend(handles=handles, loc="upper right", fontsize=fontsize, title=title)

def apply_layout_for_legend(fig, position: str):
    if position == "right":
        fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    elif position == "top":
        fig.tight_layout(rect=[0.0, 0.12, 1.0, 0.88])
    else:
        fig.tight_layout()

# ==================== Utils ====================
def time_to_seconds(t):
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
    d = df.copy()
    if treinos and "Treino" in d.columns:
        tr_norm = [str(t) for t in treinos]
        d = d[d["Treino"].astype(str).isin(tr_norm)]
    if "Out" in d.columns:
        if isinstance(out_choice, str) and out_choice == "All":
            pass
        else:
            if isinstance(out_choice, Iterable) and not isinstance(out_choice, (str, bytes)):
                outs_set = set(int(x) for x in out_choice)
                d = d[d["Out"].isin(outs_set)]
            else:
                d = d[d["Out"] == int(out_choice)]
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

    # Treinos por sessão
    session_treinos: Dict[str, List[str]] = {}
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

    # OUT(s) por sessão
    session_outs: Dict[str, List[int]] = {}
    for s in sessions:
        if "Out" in sheets[s].columns:
            outs = sorted(pd.Series(sheets[s]["Out"]).dropna().astype(int).unique().tolist())
        else:
            outs = []
        session_outs[s] = st.multiselect(
            f"OUT(s) — {s}",
            options=outs,
            default=outs,  # todos por padrão
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

    # ---------------- Gráfico principal ----------------
    series_x, series_y, labels = [], [], []
    filtered_exports = {}
    outs_sem_dados = []

    for s in sessions:
        outs_chosen = session_outs.get(s) or "All"
        df_f = get_filtered(sheets[s], outs_chosen, min_lap, max_lap, is_time_metric, treinos=session_treinos.get(s))

        if metric not in df_f.columns or df_f.empty:
            # Se ficar vazio, marcar todos OUTs selecionados como sem dados
            if isinstance(outs_chosen, list):
                for o in outs_chosen:
                    outs_sem_dados.append(f"{s} — Out {o}")
            elif outs_chosen != "All":
                outs_sem_dados.append(f"{s} — Out {outs_chosen}")
            continue

        if isinstance(outs_chosen, list) and len(outs_chosen) > 1:
            for out_n in outs_chosen:
                df_o = df_f[df_f["Out"] == out_n]
                if df_o.empty:
                    outs_sem_dados.append(f"{s} — Out {out_n}")
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
            if outs_chosen == "All" or outs_chosen == []:
                labels.append(f"{s} — Out(s): Todos")
            else:
                only = outs_chosen[0] if isinstance(outs_chosen, list) else outs_chosen
                labels.append(f"{s} — Out {int(only)}")

        filtered_exports[s] = df_f.copy()

    if outs_sem_dados:
        st.info("Sem dados após filtros para: " + "; ".join(outs_sem_dados))

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
    else:
        for x, y, lbl in zip(series_x, series_y, labels):
            if len(x) and len(y):
                ax.scatter(x, y, s=10, label=lbl)
        ax.legend(loc="upper right", fontsize="xx-small")
        ax.set_xlabel("Lap" if x_axis_mode == "Lap" else "Amostra")
        ax.set_ylabel(ylabel)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Estatísticas
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

    # Downloads
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

    # ==================== Boxplot Independente (restaurado) ====================
    st.subheader("📦 Boxplot — Seletor independente (por sessão/Out)")

    legend_pos = st.selectbox(
        "Posição da legenda (sessões)",
        ["Fora à direita", "Acima (em linha)", "Ocultar"],
        index=0,
        help="Se houver muitos carros, use 'Fora à direita' ou 'Acima (em linha)'."
    )
    legend_choice = {"Fora à direita": "right", "Acima (em linha)": "top", "Ocultar": "hide"}[legend_pos]

    all_session_names = list(sheets.keys())
    sel_sessions2 = st.multiselect(
        "Selecione sessões para análise (Boxplot independ.)",
        options=all_session_names,
        default=sessions if sessions else (default_p1[:2] if default_p1 else all_session_names[:2]),
        key="sessions_box2"
    )
    if not sel_sessions2:
        st.info("Sem dados para o boxplot independente.")
        return

    first_df2 = sheets[sel_sessions2[0]]
    time_cols2 = [c for c in first_df2.columns if c.lower().endswith("tm") and c != "SSTRAP Tm"]
    metric_opts2 = list(time_cols2)
    if "SSTRAP" in first_df2.columns:
        metric_opts2 += ["SSTRAP"]
    if not metric_opts2:
        st.warning("A sessão selecionada não tem colunas de tempo (*Tm) nem 'SSTRAP'.")
        return

    labels_map2 = {c: c for c in metric_opts2}
    if "SSTRAP" in labels_map2:
        labels_map2["SSTRAP"] = "Velocidade Máxima (SSTRAP)"
    metric2 = st.selectbox("Selecione métrica (Boxplot independ.)", options=metric_opts2,
                           format_func=lambda x: labels_map2[x], key="metric_box2")

    min_lap2 = float_input("Excluir voltas com 'Lap Tm' abaixo de (s) (valor mínimo)", default=min_lap, key="minlap_box2")
    max_lap2 = float_input("Excluir voltas com 'Lap Tm' acima de (s)", default=max_lap, key="maxlap_box2")
    if max_lap2 < min_lap2:
        st.warning("No Boxplot, o máximo não pode ser menor que o mínimo. Ajustei para o mínimo.")
        max_lap2 = float(min_lap2)

    # OUTs por sessão para a seção independente
    sel_outs_per_session = {}
    with st.container():
        st.markdown("**Selecione Out(s) (Boxplot independ.) por sessão:**")
        for idx, s in enumerate(sel_sessions2):
            df_s = sheets[s]
            outs_s = sorted(pd.Series(df_s["Out"]).dropna().unique()) if "Out" in df_s.columns else []
            default_outs = session_outs.get(s, outs_s)
            sel = st.multiselect(f"{s} — Out(s)", options=outs_s, default=default_outs, key=f"outs_box2_{idx}")
            sel_outs_per_session[s] = sel if sel else outs_s

    # montar grupos
    ys_list2, lbls2, box_sessions2 = [], [], []
    for s in sel_sessions2:
        df_s = sheets[s].copy()
        if "Lap Tm" in df_s.columns:
            df_s = df_s[pd.to_numeric(df_s["Lap Tm"], errors="coerce").notna()]
            df_s = df_s[df_s["Lap Tm"].between(float(min_lap2), float(max_lap2), inclusive="both")]
        if metric2 not in df_s.columns:
            st.warning(f"Métrica '{metric2}' não encontrada em {s}. Pulando.")
            continue

        outs_to_use = sel_outs_per_session.get(s, [])
        if not outs_to_use and "Out" in df_s.columns:
            outs_to_use = sorted(pd.Series(df_s["Out"]).dropna().unique())

        def take_values(df_g):
            return pd.to_numeric(df_g[metric2], errors="coerce").dropna().tolist()

        if "Out" not in df_s.columns or not outs_to_use:
            y = take_values(df_s)
            if y:
                ys_list2.append(y); lbls2.append(f"{s}"); box_sessions2.append(s)
        else:
            for out_n in outs_to_use:
                df_g = df_s[df_s["Out"] == out_n]
                y = take_values(df_g)
                if not y:
                    continue
                ys_list2.append(y); lbls2.append(f"{s} — Out {int(out_n)}"); box_sessions2.append(s)

    st.divider()
    st.markdown("#### 📊 Boxplot (Independente por sessão/Out)")

    if not ys_list2:
        st.info("Sem dados para o boxplot com os filtros atuais.")
        return

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    bp2 = ax2.boxplot(ys_list2, patch_artist=True)

    cycle = plt.rcParams.get("axes.prop_cycle", None)
    base_colors = cycle.by_key().get("color", ["C0"]) if cycle else ["C0"]

    present_sessions_order = []
    for s in sel_sessions2:
        if s in box_sessions2 and s not in present_sessions_order:
            present_sessions_order.append(s)
    session_to_color = {s: base_colors[i % len(base_colors)] for i, s in enumerate(present_sessions_order)}

    n_boxes2 = len(lbls2); h_in2 = fig2.get_size_inches()[1]
    fs2 = max(6, min(12, (10 * (h_in2 / 4.0)) * (8 / max(6, n_boxes2))))
    all_vals2 = np.concatenate([np.array(v, dtype=float) for v in ys_list2])
    y_range2 = float(np.nanmax(all_vals2) - np.nanmin(all_vals2)) if all_vals2.size else 1.0
    dy2 = max(0.002 * y_range2, 0.0005)

    for i, box in enumerate(bp2["boxes"]):
        sess = box_sessions2[i]
        col = session_to_color.get(sess, base_colors[0])
        box.set_facecolor(col); box.set_edgecolor(col)
        bp2["whiskers"][2*i].set_color(col); bp2["whiskers"][2*i + 1].set_color(col)
        bp2["caps"][2*i].set_color(col); bp2["caps"][2*i + 1].set_color(col)
        annotate_box(ax2, bp2, ys_list2, i, col, fs2, dy2)

    handles2 = [mpatches.Patch(facecolor=session_to_color[s], edgecolor=session_to_color[s], label=s)
                for s in present_sessions_order]

    _ = add_session_legend(ax2, handles=handles2, position=legend_choice, title="Sessões", fontsize="x-small")

    ax2.set_xticks([])
    ax2.set_xlabel("Grupos (Sessão — Out)")
    ax2.set_ylabel(labels_map2[metric2])
    apply_layout_for_legend(fig2, legend_choice)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    # ==================== GRÁFICO DE PONTOS — APENAS MÉDIA ====================
    st.subheader("📈 Média por grupo (mesma ordem do Boxplot) — cores por sessão")

    means = [float(pd.Series(pd.to_numeric(v, errors="coerce")).dropna().mean()) if len(v) else np.nan for v in ys_list2]
    x = np.arange(1, len(lbls2) + 1, dtype=float)
    fig3, ax3 = plt.subplots(figsize=(10, 4))

    vals = [v for v in means if not np.isnan(v)]
    y_range = (max(vals) - min(vals)) if vals else 1.0
    dy = max(0.002 * y_range, 0.0005)

    for i, sess in enumerate(box_sessions2):
        col = session_to_color.get(sess, "C0")
        if not np.isnan(means[i]):
            ax3.scatter([x[i]], [means[i]], marker="o", s=60, color=col, zorder=3)
            ax3.text(x[i], means[i] + dy, f"{means[i]:.3f}", ha="center", va="bottom",
                     fontsize=8, color="black",
                     bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.6, linewidth=0),
                     clip_on=True, zorder=4)

    _ = add_session_legend(ax3, handles=handles2, position=legend_choice, title="Sessões", fontsize="x-small")

    shape_handles = [Line2D([0], [0], marker="o", linestyle="None", label="Média")]
    ax3.legend(handles=shape_handles, loc="lower right", fontsize="x-small", title="Estatística")

    ax3.set_xlim(0.5, len(x) + 0.5)
    ax3.set_xticks([])
    ax3.set_xlabel("Grupos (mesma ordem do Boxplot)")
    ax3.set_ylabel(labels_map2[metric2])
    ax3.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)

    apply_layout_for_legend(fig3, legend_choice)
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

if __name__ == "__main__":
    main()
