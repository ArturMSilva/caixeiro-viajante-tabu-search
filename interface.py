"""
interface.py
------------

Interface interativa em Streamlit para explorar a execução da Busca Tabu.

Executar com:
    streamlit run interface.py

O que a interface oferece (alternativa ao GIF, mais rica):
    - Campos para configurar número de cidades, iterações e tamanho da lista tabu.
    - Botão para rodar a busca tabu.
    - Slider para navegar pelas iterações do histórico, atualizando o plot da
      rota em tempo real.
    - Gráfico de convergência (distância x iteração).
    - Distância inicial e melhor distância encontrada.

Observação de arquitetura: este arquivo consome as MESMAS funções usadas pelo
`main.py` (`tsp.py` para a lógica, `visualizacao.py` para os gráficos). Nenhuma
regra do algoritmo é reimplementada aqui.
"""

import streamlit as st

from tsp import busca_tabu, gerar_cidades, matriz_distancias
from visualizacao import plot_convergencia, plot_rota

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(page_title="TSP com Busca Tabu", page_icon="🗺️",
                   layout="wide")

st.title("🗺️ Caixeiro Viajante (TSP) com Busca Tabu")
st.caption(
    "Trabalho de Tópicos Especiais em Computação — Implementação em "
    "Otimização Combinatória (IFPI, Campus Picos)"
)

# ---------------------------------------------------------------------------
# Barra lateral: parâmetros do experimento
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Parâmetros")

    n_cidades = st.number_input(
        "Número de cidades", min_value=4, max_value=120, value=25, step=1,
        help="Quantidade de cidades sorteadas no plano.",
    )
    iteracoes = st.number_input(
        "Número de iterações", min_value=10, max_value=3000, value=300, step=10,
        help="Critério de parada da busca tabu.",
    )
    tamanho_tabu = st.number_input(
        "Tamanho da lista tabu", min_value=1, max_value=200, value=15, step=1,
        help="Por quantas iterações um movimento (swap) fica proibido.",
    )
    seed = st.number_input(
        "Seed (reprodutibilidade)", min_value=0, max_value=10_000, value=42,
        step=1,
        help="Mesma seed => mesma instância e mesma rota inicial.",
    )

    rodar = st.button("▶️ Rodar Busca Tabu", type="primary",
                      use_container_width=True)

    st.markdown("---")
    st.markdown(
        "**Como ler os resultados**\n\n"
        "- A curva azul (rota atual) *oscila*: a busca tabu aceita pioras "
        "para escapar de ótimos locais.\n"
        "- A curva vermelha (melhor global) nunca sobe.\n"
        "- Vizinhança: troca (*swap*) de pares de posições da rota."
    )

# ---------------------------------------------------------------------------
# Execução da busca
# ---------------------------------------------------------------------------
# `st.session_state` preserva o resultado entre as interações do usuário. Sem
# isso, cada movimento do slider dispararia um rerun do script e a busca seria
# executada novamente do zero.
if rodar:
    with st.spinner("Executando a Busca Tabu..."):
        cidades = gerar_cidades(int(n_cidades), seed=int(seed))
        dist = matriz_distancias(cidades)
        melhor_rota, melhor_distancia, historico = busca_tabu(
            cidades,
            dist,
            iteracoes=int(iteracoes),
            tamanho_tabu=int(tamanho_tabu),
            seed=int(seed),
        )

    st.session_state["resultado"] = {
        "cidades": cidades,
        "melhor_rota": melhor_rota,
        "melhor_distancia": melhor_distancia,
        "historico": historico,
        "parametros": {
            "cidades": int(n_cidades),
            "iteracoes": int(iteracoes),
            "tabu": int(tamanho_tabu),
            "seed": int(seed),
        },
    }

# Nada executado ainda: mostra instrução e encerra o script.
if "resultado" not in st.session_state:
    st.info("Configure os parâmetros na barra lateral e clique em "
            "**▶️ Rodar Busca Tabu**.")
    st.stop()

resultado = st.session_state["resultado"]
cidades = resultado["cidades"]
historico = resultado["historico"]
melhor_rota = resultado["melhor_rota"]
melhor_distancia = resultado["melhor_distancia"]
distancia_inicial = historico[0]["distancia"]

# ---------------------------------------------------------------------------
# Métricas principais
# ---------------------------------------------------------------------------
melhoria = 100.0 * (distancia_inicial - melhor_distancia) / distancia_inicial

col1, col2, col3 = st.columns(3)
col1.metric("Distância inicial", f"{distancia_inicial:.2f}")
col2.metric("Melhor distância encontrada", f"{melhor_distancia:.2f}",
            delta=f"-{distancia_inicial - melhor_distancia:.2f}",
            delta_color="inverse")
col3.metric("Melhoria", f"{melhoria:.2f}%")

st.markdown("---")

# ---------------------------------------------------------------------------
# Navegação pelas iterações + plot da rota
# ---------------------------------------------------------------------------
esquerda, direita = st.columns([1, 1])

with esquerda:
    st.subheader("Evolução da rota")

    ultima_iteracao = historico[-1]["iteracao"]
    iteracao_escolhida = st.slider(
        "Iteração", min_value=0, max_value=int(ultima_iteracao),
        value=int(ultima_iteracao), step=1,
        help="Arraste para percorrer o histórico da busca.",
    )

    # O histórico é indexado por posição, e como guardamos um registro por
    # iteração (começando na 0), posição == número da iteração.
    registro = historico[iteracao_escolhida]

    # Limites fixos dos eixos: evita que o gráfico mude de escala ao arrastar
    # o slider, o que dificultaria comparar visualmente as rotas.
    margem = 0.05
    x_min, x_max = cidades[:, 0].min(), cidades[:, 0].max()
    y_min, y_max = cidades[:, 1].min(), cidades[:, 1].max()
    folga_x = (x_max - x_min) * margem or 1.0
    folga_y = (y_max - y_min) * margem or 1.0
    limites = (x_min - folga_x, x_max + folga_x,
               y_min - folga_y, y_max + folga_y)

    fig_rota = plot_rota(
        cidades,
        registro["rota"],
        registro["distancia"],
        titulo=f"Iteração {registro['iteracao']} "
               f"(melhor até aqui: {registro['melhor_distancia']:.2f})",
        caminho_arquivo=None,  # None => devolve a figura em vez de salvar
        limites=limites,
    )
    st.pyplot(fig_rota, clear_figure=True)

with direita:
    st.subheader("Melhor rota encontrada")
    fig_melhor = plot_rota(
        cidades,
        melhor_rota,
        melhor_distancia,
        titulo="Melhor rota global",
        caminho_arquivo=None,
        limites=limites,
    )
    st.pyplot(fig_melhor, clear_figure=True)

# ---------------------------------------------------------------------------
# Convergência
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Convergência")
fig_conv = plot_convergencia(historico, caminho_arquivo=None)
st.pyplot(fig_conv, clear_figure=True)

# ---------------------------------------------------------------------------
# Detalhes da execução
# ---------------------------------------------------------------------------
with st.expander("🔍 Detalhes da execução"):
    st.write("**Parâmetros usados:**", resultado["parametros"])
    st.write("**Melhor rota (ordem de visita):**")
    st.code(" → ".join(str(cidade) for cidade in melhor_rota)
            + f" → {melhor_rota[0]}")
    st.write(
        f"Iterações registradas no histórico: **{len(historico)}** "
        "(inclui a iteração 0, que é a solução inicial aleatória)."
    )
