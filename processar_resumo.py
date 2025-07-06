import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta

# ==========================================
# CONFIGURAÇÃO DE CAMINHOS
# ==========================================
caminho_base = r'C:\Users\eduro\Documents\Documentos Dudu\Desafio Grao Direto\RPA - Resumo Semanal'
caminho_entradas = os.path.join(caminho_base, 'Entradas')
caminho_saida_base = os.path.join(caminho_base, 'Saida')

csv_files = [f for f in os.listdir(caminho_entradas) if f.endswith('.csv')]

# ==========================================
# LOOP PARA PROCESSAR TODOS OS CSVs
# ==========================================
for idx, csv_file in enumerate(csv_files, 1):
    caminho_csv = os.path.join(caminho_entradas, csv_file)
    nome_base = os.path.splitext(csv_file)[0]

    print(f"\n🔄 [{idx}/{len(csv_files)}] Carregando {csv_file} ...")

    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_csv_auto('{caminho_csv}', HEADER=True)").df()

    if "data" not in df.columns:
        print(f"⚠️ Coluna 'data' não encontrada em {csv_file}. Pulando.")
        continue

    df["data"] = pd.to_datetime(df["data"], errors='coerce')
    df = df.dropna(subset=["data"])
    df = df[df["data"] <= datetime.today()]

    for col in ["custo", "cliques", "leads"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df["trimestre"] = df["data"].dt.quarter
    df["ano"] = df["data"].dt.year
    df["segunda_semana"] = df["data"] - pd.to_timedelta(df["data"].dt.weekday, unit='d')
    semanas = sorted(df["segunda_semana"].dropna().unique())

    # ==========================================
    # PIPELINE GERAL
    # ==========================================
    caminho_saida_geral = os.path.join(caminho_saida_base, nome_base, "Geral")
    os.makedirs(caminho_saida_geral, exist_ok=True)
    con.execute("DROP VIEW IF EXISTS campanhas;")
    con.execute("DROP TABLE IF EXISTS campanhas;")
    con.register("campanhas", df)

    print("⚙️ Executando análises gerais...")

    # CPL + Taxas Conversão Geral
    cpl_df = con.execute("""
    SELECT canal, 
           SUM(custo) AS custo_total, 
           SUM(leads) AS leads_total, 
           CASE WHEN SUM(leads)=0 THEN 0 ELSE SUM(custo)/SUM(leads) END AS CPL,
           SUM(ativacoes) AS total_ativacoes,
           SUM(cliques) AS total_cliques,
           CASE WHEN SUM(cliques)=0 THEN 0 ELSE SUM(ativacoes)/SUM(cliques) END AS taxa_conv_ativacoes_cliques,
           SUM(cadastros) AS total_cadastros,
           CASE WHEN SUM(cadastros)=0 THEN 0 ELSE SUM(ativacoes)/SUM(cadastros) END AS taxa_conv_ativacoes_cadastros
    FROM campanhas
    GROUP BY canal
    """).df()
    cpl_df.to_csv(f"{caminho_saida_geral}\\cpl_taxas_conversao_por_canal.csv", index=False)

    # Melhor grupo Q1 2025
    grupo_df = con.execute("""
    SELECT grupo_anuncio,
           SUM(cadastros) AS total_cadastros,
           SUM(ativacoes) AS total_ativacoes,
           CASE WHEN SUM(cadastros)=0 THEN 0 ELSE SUM(ativacoes)/SUM(cadastros) END AS taxa_conversao
    FROM campanhas
    WHERE trimestre = 1 AND ano = 2025
    GROUP BY grupo_anuncio
    ORDER BY taxa_conversao DESC
    LIMIT 1
    """).df()
    grupo_df.to_csv(f"{caminho_saida_geral}\\melhor_grupo_q1_2025.csv", index=False)

    # CPL médio por canal e trimestre
    clp_df = con.execute("""
    SELECT canal, 
           trimestre, 
           SUM(custo) AS custo_total, 
           SUM(leads) AS leads_total, 
           CASE WHEN SUM(leads)=0 THEN 0 ELSE SUM(custo)/SUM(leads) END AS cpl_medio
    FROM campanhas
    GROUP BY canal, trimestre
    HAVING SUM(leads) > 300
    ORDER BY canal, trimestre
    """).df()
    clp_df.to_csv(f"{caminho_saida_geral}\\clp_medio_canal_trimestre.csv", index=False)

    # Top 5 grupos taxa de conversão
    top5_df = con.execute("""
    SELECT grupo_anuncio,
           SUM(ativacoes) AS total_ativacoes,
           SUM(cadastros) AS total_cadastros,
           CASE WHEN SUM(cadastros)=0 THEN 0 ELSE SUM(ativacoes)/SUM(cadastros) END AS taxa_conversao
    FROM campanhas
    GROUP BY grupo_anuncio
    ORDER BY taxa_conversao DESC
    LIMIT 5
    """).df()
    top5_df.to_csv(f"{caminho_saida_geral}\\top5_grupos_taxa_conversao.csv", index=False)

    # Evolução mensal taxa de cliques
    evolucao_df = con.execute("""
    SELECT STRFTIME(data, '%Y-%m') AS ano_mes,
           canal,
           SUM(cliques) AS total_cliques,
           SUM(impressoes) AS total_impressoes,
           CASE WHEN SUM(impressoes)=0 THEN 0 ELSE SUM(cliques)/SUM(impressoes) END AS taxa_clique
    FROM campanhas
    GROUP BY ano_mes, canal
    ORDER BY ano_mes ASC, canal
    """).df()
    evolucao_df.to_csv(f"{caminho_saida_geral}\\evolucao_mensal_taxa_clique.csv", index=False)

    print("✅ Análises gerais concluídas e salvas.\n")

    # ==========================================
    # GRÁFICOS GERAIS - Evolução e Tendência de Leads por Canal
    # ==========================================
    print("📊 Gerando gráficos gerais de evolução e tendência de leads...")

    # Agrupa dados gerais por data e canal
    leads_tempo = df.groupby(["data", "canal"])["leads"].sum().reset_index()

    # Cria coluna numérica para regressão
    leads_tempo["data_num"] = leads_tempo["data"].map(lambda x: x.toordinal())

    # ---------------------------------------
    # Gráfico 1: Pontos reais por canal
    # ---------------------------------------
    plt.figure(figsize=(12, 6))

    for canal in leads_tempo["canal"].unique():
        df_canal = leads_tempo[leads_tempo["canal"] == canal]
        plt.plot(df_canal["data"], df_canal["leads"], marker="o", linestyle="-", label=canal)

    plt.title("Evolução de Leads por Canal ao Longo do Tempo")
    plt.xlabel("Data")
    plt.ylabel("Número de Leads")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{caminho_saida_geral}\\evolucao_leads_canal_geral.png")
    plt.close()

    # ---------------------------------------
    # Gráfico 2: Regressão linear por canal
    # ---------------------------------------
    plt.figure(figsize=(12, 6))

    for canal in leads_tempo["canal"].unique():
        df_canal = leads_tempo[leads_tempo["canal"] == canal]
        coef = np.polyfit(df_canal["data_num"], df_canal["leads"], 1)
        poly1d_fn = np.poly1d(coef)
        plt.plot(
            df_canal["data"],
            poly1d_fn(df_canal["data_num"]),
            linestyle="-",
            label=f"Tendência - {canal}"
        )

    plt.title("Tendência de Leads por Canal ao Longo do Tempo (Regressão Linear)")
    plt.xlabel("Data")
    plt.ylabel("Número de Leads (Tendência)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{caminho_saida_geral}\\tendencia_leads_canal_geral.png")
    plt.close()

    print("✅ Gráficos gerais de evolução e tendência gerados e salvos.\n")

    # ==========================================
    # GRÁFICO GERAL - CPL Médio por Trimestre e Canal
    # ==========================================
    print("📊 Gerando gráfico CPL médio por trimestre e canal (geral)...")

    # Agrupar dados por trimestre e canal
    cpl_trimestre = df.groupby(["trimestre", "canal"]).agg({"custo": "sum", "leads": "sum"}).reset_index()

    # Calcular CPL
    cpl_trimestre["CPL"] = cpl_trimestre["custo"] / cpl_trimestre["leads"].replace(0, np.nan)
    cpl_trimestre["CPL"] = cpl_trimestre["CPL"].fillna(0)

    # Gerar gráfico
    plt.figure(figsize=(10, 5))
    import seaborn as sns
    sns.barplot(data=cpl_trimestre, x="trimestre", y="CPL", hue="canal")

    # Personalização
    plt.title("CPL Médio por Trimestre e Canal")
    plt.xlabel("Trimestre")
    plt.ylabel("CPL Médio")
    plt.grid(axis="y")
    plt.legend(title="Canal")
    plt.tight_layout()

    # Salvar PNG
    plt.savefig(f"{caminho_saida_geral}\\cpl_trimestre_canal_geral.png")
    plt.close()

    print("✅ Gráfico CPL médio por trimestre e canal gerado e salvo.\n")

    # ==========================================
    # GRÁFICO GERAL - Tendência de Ativações desde 2023
    # ==========================================
    print("📊 Gerando gráfico de tendência de ativações desde 2023 (geral)...")

    # Filtrar dados a partir de 2023
    dados_2023 = df[df["data"].dt.year >= 2023].copy()

    if not dados_2023.empty and "ativacoes" in dados_2023.columns:
        # Criar coluna ano-mês
        dados_2023["ano_mes"] = dados_2023["data"].dt.to_period("M").astype(str)

        # Agrupar por ano-mês somando ativações
        ativacoes_tendencia = dados_2023.groupby("ano_mes")["ativacoes"].sum().reset_index()

        # Criar coluna numérica sequencial para regressão
        ativacoes_tendencia["mes_num"] = np.arange(len(ativacoes_tendencia))

        # Calcular regressão linear
        coef = np.polyfit(ativacoes_tendencia["mes_num"], ativacoes_tendencia["ativacoes"], 1)
        trend = np.poly1d(coef)

        # Gerar gráfico
        plt.figure(figsize=(10, 5))

        # Linha real
        import seaborn as sns
        sns.lineplot(data=ativacoes_tendencia, x="ano_mes", y="ativacoes", marker="o", label="Ativações")

        # Linha tendência
        plt.plot(
            ativacoes_tendencia["ano_mes"],
            trend(ativacoes_tendencia["mes_num"]),
            color="red",
            linestyle="-",
            label="Tendência"
        )

        # Personalização
        plt.title("Tendência de Ativações de Campanhas desde 2023")
        plt.xlabel("Ano-Mês")
        plt.ylabel("Número de Ativações")
        plt.xticks(rotation=65)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Salvar PNG
        plt.savefig(f"{caminho_saida_geral}\\tendencia_ativacoes_2023_geral.png")
        plt.close()

        print("✅ Gráfico de tendência de ativações desde 2023 gerado e salvo.\n")
    else:
        print("⚠️ Sem dados de ativações para gerar gráfico de tendência desde 2023.\n")

    
    # ==========================================
    # Trimestres com desempenhos semelhantes entre Google Ads e Meta Ads
    # ==========================================
    print("📊 Gerando CSV de trimestres com desempenhos semelhantes entre Google Ads e Meta Ads...")

    # Garantir que a coluna ano existe
    df["ano"] = df["data"].dt.year

    # Agrupar por ano, trimestre e canal
    leads_trimestre = df.groupby(["ano", "trimestre", "canal"])["leads"].sum().reset_index()

    # Pivotar para colunas separadas Google e Meta
    leads_pivot = leads_trimestre.pivot(index=["ano", "trimestre"], columns="canal", values="leads").reset_index()

    # Calcular diferença absoluta
    leads_pivot["diferença"] = abs(leads_pivot["Google Ads"] - leads_pivot["Meta Ads"])

    # Definir tolerância e filtrar
    tolerancia = 500
    trimestres_semelhantes = leads_pivot[leads_pivot["diferença"] <= tolerancia]

    # Salvar CSV
    trimestres_semelhantes.to_csv(f"{caminho_saida_geral}\\trimestres_semelhantes_google_meta.csv", index=False)

    print("✅ CSV de trimestres semelhantes gerado e salvo.\n")


    # ==========================================
    # PIPELINE SEMANAL
    # ==========================================
    print("⚙️ Executando análises semanais...")

    for i, semana in enumerate(semanas):
        try:
            semana = pd.Timestamp(semana)
            inicio_semana = semana
            fim_semana = semana + timedelta(days=6)
            nome_pasta_semana = semana.strftime('%Y-%m-%d')

            caminho_saida_semana = os.path.join(caminho_saida_base, nome_base, nome_pasta_semana)
            os.makedirs(caminho_saida_semana, exist_ok=True)

            mask = (df["data"].dt.date >= inicio_semana.date()) & (df["data"].dt.date <= fim_semana.date())
            df_semana = df[mask]
            print(f"✅ Semana {nome_pasta_semana}: {len(df_semana)} registros encontrados.")

            if df_semana.empty:
                print(f"⚠️ Sem dados para {nome_pasta_semana}. Pulando.")
                continue

            con.execute("DROP VIEW IF EXISTS campanhas;")
            con.execute("DROP TABLE IF EXISTS campanhas;")
            con.register("campanhas", df_semana)

            # =====================================================
            # Gráfico semanal: Evolução de Leads por Canal
            # =====================================================
            leads_tempo = df_semana.groupby(["data", "canal"])["leads"].sum().reset_index()
            plt.figure(figsize=(12, 6))
            for canal in leads_tempo["canal"].unique():
                canal_data = leads_tempo[leads_tempo["canal"] == canal]
                plt.plot(canal_data["data"], canal_data["leads"], marker='o', label=canal)
            plt.title(f"Evolução de Leads por Canal - {nome_pasta_semana}")
            plt.xlabel("Data")
            plt.ylabel("Leads")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{caminho_saida_semana}\\evolucao_leads_canal.png")
            plt.close()

            # =====================================================
            # Gráfico semanal: CPL médio por canal
            # =====================================================
            cpl_trimestre = df_semana.groupby(["canal"]).agg({"custo": "sum", "leads": "sum"}).reset_index()
            cpl_trimestre["CPL"] = cpl_trimestre["custo"] / cpl_trimestre["leads"].replace(0, np.nan)
            cpl_trimestre["CPL"] = cpl_trimestre["CPL"].fillna(0)
            plt.figure(figsize=(10, 5))
            plt.bar(cpl_trimestre["canal"], cpl_trimestre["CPL"])
            plt.title(f"CPL Médio por Canal - {nome_pasta_semana}")
            plt.xlabel("Canal")
            plt.ylabel("CPL")
            plt.xticks(rotation=45)
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.savefig(f"{caminho_saida_semana}\\cpl_medio_canal.png")
            plt.close()

            # =====================================================
            # Comparação com semana anterior (em CSV)
            # =====================================================
            if i > 0:
                semana_anterior = pd.Timestamp(semanas[i - 1])
                inicio_ant = semana_anterior
                fim_ant = semana_anterior + timedelta(days=6)
                df_ant = df[(df["data"].dt.date >= inicio_ant.date()) & (df["data"].dt.date <= fim_ant.date())]

                if not df_ant.empty:
                    comparativo = pd.DataFrame({
                        "Métrica": ["Leads", "Ativações", "Cliques", "Cadastros", "CPL"],
                        "Semana Atual": [
                            df_semana["leads"].sum(),
                            df_semana["ativacoes"].sum(),
                            df_semana["cliques"].sum(),
                            df_semana["cadastros"].sum(),
                            df_semana["custo"].sum() / df_semana["leads"].sum() if df_semana["leads"].sum() != 0 else 0
                        ],
                        "Semana Anterior": [
                            df_ant["leads"].sum(),
                            df_ant["ativacoes"].sum(),
                            df_ant["cliques"].sum(),
                            df_ant["cadastros"].sum(),
                            df_ant["custo"].sum() / df_ant["leads"].sum() if df_ant["leads"].sum() != 0 else 0
                        ]
                    })
                    comparativo.to_csv(f"{caminho_saida_semana}\\comparativo_semana_anterior.csv", index=False)

            # =====================================================
            # Contexto semanal para preencher documento no UiPath
            # =====================================================
            print(f"🔍 Gerando contexto de resumo para semana {nome_pasta_semana}...")

            contexto_semana = {}

            # Melhor CPL
            canal_melhor_row = cpl_trimestre.loc[cpl_trimestre["CPL"].idxmin()]
            contexto_semana["canal_melhor_cpl"] = canal_melhor_row["canal"]
            contexto_semana["valor_melhor_cpl"] = round(canal_melhor_row["CPL"], 2)

            # Total de leads
            total_leads = df_semana["leads"].sum()
            contexto_semana["total_leads"] = int(total_leads)

            # Variação de leads
            if i > 0 and not df_ant.empty:
                leads_ant = df_ant["leads"].sum()
                variacao_leads = ((total_leads - leads_ant) / leads_ant * 100) if leads_ant != 0 else 0
            else:
                variacao_leads = 0
            contexto_semana["variacao_leads"] = f"{variacao_leads:.2f}%"

            # Taxa de conversão geral
            ativacoes = df_semana["ativacoes"].sum()
            taxa_conv_geral = (ativacoes / total_leads * 100) if total_leads != 0 else 0
            contexto_semana["taxa_conversao_geral"] = f"{taxa_conv_geral:.2f}"

            # Google Ads
            google = df_semana[df_semana["canal"] == "Google Ads"]
            cpl_google = google["custo"].sum() / google["leads"].sum() if google["leads"].sum() != 0 else 0
            taxa_conv_google = (google["ativacoes"].sum() / google["leads"].sum() * 100) if google["leads"].sum() != 0 else 0
            contexto_semana["cpl_google"] = round(cpl_google, 2)
            contexto_semana["taxa_conversao_google"] = f"{taxa_conv_google:.2f}"
            contexto_semana["leads_google"] = int(google["leads"].sum())

            # Meta Ads
            meta = df_semana[df_semana["canal"] == "Meta Ads"]
            cpl_meta = meta["custo"].sum() / meta["leads"].sum() if meta["leads"].sum() != 0 else 0
            taxa_conv_meta = (meta["ativacoes"].sum() / meta["leads"].sum() * 100) if meta["leads"].sum() != 0 else 0
            contexto_semana["cpl_meta"] = round(cpl_meta, 2)
            contexto_semana["taxa_conversao_meta"] = f"{taxa_conv_meta:.2f}"
            contexto_semana["leads_meta"] = int(meta["leads"].sum())

            # Campanha com mais leads
            if "campanha" in df_semana.columns:
                campanha_top_row = df_semana.groupby("campanha")["leads"].sum().reset_index().sort_values(by="leads", ascending=False).iloc[0]
                contexto_semana["campanha_top_leads"] = campanha_top_row["campanha"]
                contexto_semana["leads_campanha_top"] = int(campanha_top_row["leads"])
            else:
                contexto_semana["campanha_top_leads"] = "N/A"
                contexto_semana["leads_campanha_top"] = 0

            # Grupo de anúncio com maior taxa de conversão
            if "grupo_anuncio" in df_semana.columns and "cadastros" in df_semana.columns:
                df_semana = df_semana.copy()
                df_semana["taxa_conversao"] = df_semana.apply(lambda x: (x["ativacoes"] / x["cadastros"] * 100) if x["cadastros"] > 0 else 0,axis=1)
                grupo_top = df_semana.groupby("grupo_anuncio").agg({"ativacoes": "sum", "cadastros": "sum"}).reset_index()
                grupo_top["taxa_conversao"] = grupo_top.apply(lambda x: (x["ativacoes"] / x["cadastros"] * 100) if x["cadastros"] > 0 else 0, axis=1)
                grupo_top_row = grupo_top.loc[grupo_top["taxa_conversao"].idxmax()]
                contexto_semana["grupo_top_taxa"] = grupo_top_row["grupo_anuncio"]
                contexto_semana["taxa_grupo_top"] = f"{grupo_top_row['taxa_conversao']:.2f}"
            else:
                contexto_semana["grupo_top_taxa"] = "N/A"
                contexto_semana["taxa_grupo_top"] = "0.00"

            # Salvar contexto em CSV para UiPath
            pd.DataFrame([contexto_semana]).to_csv(f"{caminho_saida_semana}\\contexto_semana_{nome_pasta_semana}.csv", index=False, encoding="utf-8-sig")

            print(f"✅ Semana {nome_pasta_semana} processada com sucesso.\n")

        except Exception as e:
            print(f"❌ Erro ao processar a semana {nome_pasta_semana}: {e}. Pulando.\n")
            continue

    print("✅✅ TODAS AS ANÁLISES SEMANAIS FORAM CONCLUÍDAS E SALVAS ORGANIZADAS.")
