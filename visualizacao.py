"""
visualizacao.py
---------------

Camada de VISUALIZAÇÃO estática do projeto: gráficos em PNG e animação em GIF.

Este módulo depende de `tsp.py` apenas indiretamente (recebe cidades, rotas e
histórico como dados). A lógica de otimização não sabe que este arquivo existe,
o que permite trocar/remover a visualização sem tocar no algoritmo.

Todas as funções de plot aceitam `caminho_arquivo=None`. Nesse caso, em vez de
salvar em disco, a figura do matplotlib é devolvida — é assim que a interface
Streamlit (`interface.py`) reaproveita exatamente os mesmos gráficos.
"""

import os
import shutil

import matplotlib

# Backend "Agg": renderiza para arquivo/buffer sem precisar de janela gráfica.
# Essencial para rodar em terminal, servidor ou dentro do Streamlit.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (import após definir o backend)
import imageio.v2 as imageio  # noqa: E402


def plot_rota(cidades, rota, distancia, titulo="Rota", caminho_arquivo=None,
              limites=None):
    """Desenha as cidades (pontos) e a rota (linhas na ordem de visita).

    O ciclo é fechado explicitamente: repetimos a primeira cidade no final da
    sequência de coordenadas, para que o segmento de retorno também apareça.

    Parâmetros
    ----------
    cidades : np.ndarray (n, 2)
        Coordenadas das cidades.
    rota : sequência de int
        Ordem de visita (permutação dos índices das cidades).
    distancia : float
        Valor da função objetivo, exibido no título.
    titulo : str
        Texto principal do título.
    caminho_arquivo : str | None
        Se informado, salva a figura nesse caminho e fecha a figura.
        Se None, devolve o objeto Figure (usado pelo Streamlit).
    limites : tuple | None
        Opcional: (x_min, x_max, y_min, y_max). Fixar os limites dos eixos é
        importante no GIF/slider, senão cada frame "pula" de escala e a
        animação fica tremida.

    Retorna
    -------
    matplotlib.figure.Figure | None
    """
    # Coordenadas na ordem da rota, com a primeira cidade repetida no fim para
    # que o segmento de retorno (o fechamento do ciclo) também seja desenhado.
    ordem = list(rota) + [rota[0]]
    xs = cidades[ordem, 0]
    ys = cidades[ordem, 1]

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(xs, ys, "-", color="#1f77b4", linewidth=1.4, zorder=1)
    ax.plot(cidades[:, 0], cidades[:, 1], "o", color="#d62728",
            markersize=6, zorder=2)

    # Destaque da cidade de partida/chegada (ajuda a "ler" o ciclo).
    ax.plot(cidades[rota[0], 0], cidades[rota[0], 1], "s", color="#2ca02c",
            markersize=10, zorder=3, label="Cidade inicial")

    # Rótulo com o índice de cada cidade — facilita conferir a rota manualmente.
    for indice, (x, y) in enumerate(cidades):
        ax.annotate(str(indice), (x, y), textcoords="offset points",
                    xytext=(5, 4), fontsize=7, color="#444444")

    ax.set_title(f"{titulo}\nDistância total: {distancia:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="box")

    if limites is not None:
        x_min, x_max, y_min, y_max = limites
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    fig.tight_layout()

    if caminho_arquivo is None:
        return fig

    fig.savefig(caminho_arquivo, dpi=120)
    plt.close(fig)  # libera memória: importante ao gerar centenas de frames
    return None


def plot_convergencia(historico, caminho_arquivo=None):
    """Plota a evolução da distância ao longo das iterações.

    Duas curvas são desenhadas, e a comparação entre elas é o ponto didático
    mais interessante do trabalho:

    - "Rota atual": oscila para cima e para baixo, porque a busca tabu ACEITA
      pioras para escapar de ótimos locais.
    - "Melhor global": monotonicamente não-crescente (só desce), pois guarda
      a melhor solução já vista.

    Parâmetros
    ----------
    historico : list[dict]
        Saída de `tsp.busca_tabu`.
    caminho_arquivo : str | None
        Igual a `plot_rota`: salva em disco ou devolve a figura.

    Retorna
    -------
    matplotlib.figure.Figure | None
    """
    iteracoes = [registro["iteracao"] for registro in historico]
    atuais = [registro["distancia"] for registro in historico]
    melhores = [registro["melhor_distancia"] for registro in historico]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(iteracoes, atuais, color="#1f77b4", linewidth=1.0,
            label="Distância da rota atual")
    ax.plot(iteracoes, melhores, color="#d62728", linewidth=1.8,
            label="Melhor distância global")

    ax.set_title("Convergência da Busca Tabu")
    ax.set_xlabel("Iteração")
    ax.set_ylabel("Distância total")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if caminho_arquivo is None:
        return fig

    fig.savefig(caminho_arquivo, dpi=120)
    plt.close(fig)
    return None


def gerar_gif(cidades, historico, caminho_gif="evolucao.gif", passo=5,
              duracao_frame=0.18, pasta_frames="frames_temp"):
    """Gera um GIF mostrando a rota evoluindo ao longo das iterações.

    Estratégia: renderizar um PNG por frame numa pasta temporária, montar o GIF
    a partir desses arquivos e, por fim, apagar a pasta.

    O parâmetro `passo` faz subamostragem do histórico (1 frame a cada `passo`
    iterações). Sem isso, 300 iterações gerariam 300 frames e um GIF pesado e
    lento de abrir. A última iteração é sempre incluída, para que o GIF termine
    no estado final da busca.

    Parâmetros
    ----------
    cidades : np.ndarray (n, 2)
    historico : list[dict]
        Saída de `tsp.busca_tabu`.
    caminho_gif : str
        Arquivo de saída (.gif).
    passo : int
        Intervalo de subamostragem das iterações.
    duracao_frame : float
        Duração de cada frame, em segundos.
    pasta_frames : str
        Pasta temporária para os PNGs intermediários (removida no final).

    Retorna
    -------
    str
        Caminho do GIF gerado.
    """
    # Limites fixos dos eixos, com uma pequena margem: mantém a escala estável
    # entre frames para a animação não "tremer".
    margem = 0.05
    x_min, x_max = cidades[:, 0].min(), cidades[:, 0].max()
    y_min, y_max = cidades[:, 1].min(), cidades[:, 1].max()
    folga_x = (x_max - x_min) * margem or 1.0
    folga_y = (y_max - y_min) * margem or 1.0
    limites = (x_min - folga_x, x_max + folga_x,
               y_min - folga_y, y_max + folga_y)

    # Seleção dos frames: de `passo` em `passo`, garantindo o último registro
    # para que o GIF termine no estado final da busca.
    indices = list(range(0, len(historico), max(1, passo)))
    if indices[-1] != len(historico) - 1:
        indices.append(len(historico) - 1)

    # Pasta temporária começa sempre limpa (evita frames de execuções antigas).
    if os.path.isdir(pasta_frames):
        shutil.rmtree(pasta_frames)
    os.makedirs(pasta_frames, exist_ok=True)

    arquivos_frames = []
    try:
        for contador, indice in enumerate(indices):
            registro = historico[indice]
            caminho_frame = os.path.join(pasta_frames, f"frame_{contador:04d}.png")

            plot_rota(
                cidades,
                registro["rota"],
                registro["distancia"],
                titulo=(f"Busca Tabu — iteração {registro['iteracao']} "
                        f"(melhor: {registro['melhor_distancia']:.2f})"),
                caminho_arquivo=caminho_frame,
                limites=limites,
            )
            arquivos_frames.append(caminho_frame)

        # Montagem do GIF. `loop=0` = repetir indefinidamente.
        imagens = [imageio.imread(arquivo) for arquivo in arquivos_frames]
        imageio.mimsave(caminho_gif, imagens, duration=duracao_frame, loop=0)
    finally:
        # `finally` garante a limpeza mesmo se algo falhar no meio do processo.
        if os.path.isdir(pasta_frames):
            shutil.rmtree(pasta_frames, ignore_errors=True)

    return caminho_gif
