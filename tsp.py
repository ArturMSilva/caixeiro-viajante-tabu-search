"""
tsp.py
------

Núcleo lógico do projeto: modelagem do Problema do Caixeiro Viajante (TSP)
e implementação da metaheurística Busca Tabu (Tabu Search).

IMPORTANTE (separação de responsabilidades):
    Este módulo NÃO conhece nada sobre visualização (matplotlib, Streamlit, GIF).
    Ele apenas gera dados do problema, avalia soluções e devolve resultados +
    histórico. Quem quiser desenhar algo usa `visualizacao.py` ou `interface.py`.

Convenções usadas em todo o arquivo:
    - `cidades` é um array NumPy de shape (n, 2), onde cada linha é (x, y).
    - `rota` é uma lista (ou array) de índices de cidades, uma permutação de
      0..n-1. A rota é sempre interpretada como um CICLO FECHADO: depois da
      última cidade, volta-se para a primeira.
    - `dist` é a matriz de distâncias, shape (n, n), com dist[i][j] = distância
      euclidiana entre a cidade i e a cidade j.
"""

from collections import deque

import numpy as np


def gerar_cidades(n, seed=42, largura=100.0, altura=100.0):
    """Gera uma instância aleatória do TSP com `n` cidades no plano.

    Como o trabalho é de implementação (e não de estudo de instâncias reais),
    geramos as cidades aleatoriamente dentro de um retângulo
    [0, largura] x [0, altura]. O `seed` garante REPRODUTIBILIDADE: rodando
    duas vezes com o mesmo seed, a mesma instância é gerada — isso é essencial
    para comparar execuções da busca tabu com parâmetros diferentes.

    Parâmetros
    ----------
    n : int
        Quantidade de cidades.
    seed : int
        Semente do gerador pseudoaleatório.
    largura, altura : float
        Dimensões da região onde as cidades são sorteadas.

    Retorna
    -------
    np.ndarray de shape (n, 2)
        Coordenadas (x, y) de cada cidade.
    """
    # Generator moderno do NumPy: preferível ao np.random.seed() global,
    # porque não interfere no estado aleatório do resto do programa.
    rng = np.random.default_rng(seed)

    xs = rng.uniform(0.0, largura, size=n)
    ys = rng.uniform(0.0, altura, size=n)
    return np.column_stack((xs, ys))


def matriz_distancias(cidades):
    """Calcula a matriz de distâncias euclidianas entre todas as cidades.

    Pré-computar essa matriz é a otimização mais importante do projeto:
    a busca tabu avalia MUITAS soluções (centenas de vizinhos por iteração),
    e cada avaliação só precisa somar valores já calculados — nada de
    recalcular raízes quadradas a todo momento.

    A conta é feita de forma vetorizada (broadcasting):
        cidades[:, None, :] tem shape (n, 1, 2)
        cidades[None, :, :] tem shape (1, n, 2)
    A subtração gera (n, n, 2) com todas as diferenças (dx, dy); então
    aplicamos a norma euclidiana no último eixo.

    Retorna
    -------
    np.ndarray de shape (n, n), simétrica, com diagonal zero.
    """
    diferencas = cidades[:, None, :] - cidades[None, :, :]
    return np.sqrt((diferencas ** 2).sum(axis=2))


def distancia_total(rota, dist):
    """Função objetivo: comprimento total do ciclo definido por `rota`.

    Somamos as arestas consecutivas da rota e, por último, a aresta de retorno
    da cidade final para a inicial (o que fecha o ciclo — restrição do TSP).

    Implementação vetorizada:
        - `rota` é convertida em array de índices;
        - `np.roll(rota, -1)` produz a rota deslocada uma posição à esquerda,
          ou seja, para cada posição k temos o "próximo" da rota, e o próximo
          do último é justamente o primeiro (fechando o ciclo automaticamente);
        - dist[origens, destinos] usa indexação avançada para pegar todas as
          arestas de uma vez e somamos.

    Parâmetros
    ----------
    rota : sequência de int
        Permutação dos índices das cidades.
    dist : np.ndarray
        Matriz de distâncias pré-computada.

    Retorna
    -------
    float
        Distância total do ciclo fechado.
    """
    origens = np.asarray(rota, dtype=int)
    # O "próximo" do último elemento passa a ser o primeiro: o ciclo fecha sozinho.
    destinos = np.roll(origens, -1)
    return float(dist[origens, destinos].sum())


def rota_inicial(n_cidades, seed=42):
    """Gera uma solução inicial aleatória (permutação das cidades).

    A busca tabu é uma metaheurística de busca local: ela precisa de um ponto
    de partida. Usamos uma permutação puramente aleatória de propósito —
    assim fica evidente, nos gráficos e no GIF, o quanto a busca melhora a
    solução (partir de uma heurística gulosa esconderia esse efeito).

    Retorna
    -------
    list[int]
        Permutação de 0..n_cidades-1.
    """
    rng = np.random.default_rng(seed)
    rota = np.arange(n_cidades)
    rng.shuffle(rota)
    return rota.tolist()


def vizinhos_swap(rota):
    """Gera a vizinhança da rota atual por TROCA (swap) de pares de posições.

    Definição da vizinhança N(s): duas rotas são vizinhas se uma pode ser
    obtida da outra trocando de lugar as cidades que ocupam duas posições
    i e j. Para uma rota de n cidades existem C(n, 2) = n(n-1)/2 vizinhos.

    Devolvemos também QUAL movimento gerou cada vizinho — o par (i, j) —
    porque é exatamente esse par que será armazenado na lista tabu para
    proibir o movimento inverso nas próximas iterações.

    Parâmetros
    ----------
    rota : sequência de int

    Retorna
    -------
    list[tuple[list[int], tuple[int, int]]]
        Lista de pares (rota_vizinha, movimento), onde movimento = (i, j)
        com i < j (par ordenado, para que (i, j) e (j, i) sejam o MESMO
        movimento na lista tabu).
    """
    rota = list(rota)
    n = len(rota)
    vizinhanca = []

    # i < j evita gerar o mesmo vizinho duas vezes (o swap é simétrico) e faz
    # com que (i, j) e (j, i) sejam UM único movimento aos olhos da lista tabu.
    for i in range(n - 1):
        for j in range(i + 1, n):
            vizinho = rota.copy()
            vizinho[i], vizinho[j] = vizinho[j], vizinho[i]
            vizinhanca.append((vizinho, (i, j)))

    return vizinhanca


def busca_tabu(cidades, dist, iteracoes=300, tamanho_tabu=15, seed=42):
    """Executa a Busca Tabu para o TSP.

    Ideia geral da metaheurística (Glover, 1989):
        A busca local pura (descida) para no primeiro ótimo local. A busca
        tabu permite MOVIMENTOS DE PIORA para escapar desses ótimos locais,
        mas mantém uma "memória de curto prazo" — a LISTA TABU — com os
        últimos movimentos realizados, proibindo-os temporariamente. Isso
        evita que o algoritmo desfaça o que acabou de fazer e volte a ciclar
        entre as mesmas duas soluções.

    Passos executados a cada iteração:
        1. Gerar toda a vizinhança da rota atual (swaps de pares).
        2. Escolher o MELHOR vizinho admissível, onde admissível significa:
             - o movimento não está na lista tabu; OU
             - o movimento está na lista tabu, mas satisfaz o CRITÉRIO DE
               ASPIRAÇÃO (gera uma solução melhor que a melhor global já
               encontrada) — nesse caso o tabu é ignorado, pois seria
               absurdo recusar a melhor solução vista até agora.
        3. Mover-se para esse vizinho (mesmo que ele seja PIOR que a rota
           atual — é isso que dá poder de escape à busca tabu).
        4. Registrar o movimento na lista tabu.
        5. Atualizar a melhor solução global, se houve melhora.

    Critério de parada: número fixo de iterações (`iteracoes`).

    Complexidade por iteração: O(n^2) vizinhos, cada um avaliado em O(n)
    (soma das arestas do ciclo) => O(n^3) por iteração e O(iteracoes * n^3)
    no total. Ainda assim é polinomial — muito melhor que O(n!) da busca
    exaustiva —, e a avaliação vetorizada com NumPy mantém a execução rápida
    para as instâncias usadas no trabalho.

    Parâmetros
    ----------
    cidades : np.ndarray (n, 2)
        Coordenadas das cidades (usado apenas para descobrir n; a avaliação
        em si usa a matriz `dist`).
    dist : np.ndarray (n, n)
        Matriz de distâncias pré-computada.
    iteracoes : int
        Número de iterações (critério de parada).
    tamanho_tabu : int
        Tamanho (tenure) da lista tabu: por quantas iterações um movimento
        permanece proibido. Valor pequeno => risco de ciclagem; valor muito
        grande => busca engessada, pouca liberdade de movimento.
    seed : int
        Semente usada para gerar a rota inicial.

    Retorna
    -------
    melhor_rota : list[int]
        Melhor rota encontrada em toda a execução.
    melhor_distancia : float
        Distância dessa melhor rota (valor da função objetivo).
    historico : list[dict]
        Um registro por iteração (mais o estado inicial, iteração 0), com:
            'iteracao'          -> índice da iteração
            'rota'              -> rota corrente naquela iteração
            'distancia'         -> distância da rota corrente
            'melhor_distancia'  -> melhor distância global até ali
        Esse histórico é o que alimenta o gráfico de convergência, o GIF e
        o slider da interface Streamlit.
    """
    n = len(cidades)

    # --- Solução inicial ---
    rota_atual = rota_inicial(n, seed=seed)
    distancia_atual = distancia_total(rota_atual, dist)

    # A melhor solução global começa sendo a própria solução inicial.
    melhor_rota = rota_atual.copy()
    melhor_distancia = distancia_atual

    # --- Lista tabu ---
    # deque com `maxlen`: ao inserir o (tamanho_tabu + 1)-ésimo movimento, o mais
    # antigo é descartado automaticamente. É exatamente a semântica de uma memória
    # de curto prazo com tenure fixa (FIFO), sem código de expiração manual.
    lista_tabu = deque(maxlen=tamanho_tabu)

    # O histórico começa com o estado inicial (iteração 0), que alimenta os
    # gráficos, o GIF e o slider da interface.
    historico = [{
        "iteracao": 0,
        "rota": rota_atual.copy(),
        "distancia": distancia_atual,
        "melhor_distancia": melhor_distancia,
    }]

    for it in range(1, iteracoes + 1):
        # 1) Vizinhança completa da solução corrente.
        vizinhanca = vizinhos_swap(rota_atual)

        # 2) Selecionar o melhor vizinho ADMISSÍVEL.
        melhor_vizinho = None
        melhor_vizinho_distancia = float("inf")
        melhor_movimento = None

        for vizinho, movimento in vizinhanca:
            d_vizinho = distancia_total(vizinho, dist)

            eh_tabu = movimento in lista_tabu
            # Critério de aspiração: mesmo proibido, o movimento é liberado se
            # levar a uma solução melhor que a melhor global conhecida — seria
            # absurdo recusar a melhor solução já vista por causa do tabu.
            aspiracao = d_vizinho < melhor_distancia

            if eh_tabu and not aspiracao:
                continue  # proibido e sem mérito -> descartado

            if d_vizinho < melhor_vizinho_distancia:
                melhor_vizinho = vizinho
                melhor_vizinho_distancia = d_vizinho
                melhor_movimento = movimento

        # Caso extremo: todos os vizinhos estavam proibidos (acontece com
        # vizinhança pequena e lista tabu grande). Sem candidato, registramos a
        # iteração e seguimos — a lista tabu esvazia e a busca volta a ter opções.
        if melhor_vizinho is None:
            historico.append({
                "iteracao": it,
                "rota": rota_atual.copy(),
                "distancia": distancia_atual,
                "melhor_distancia": melhor_distancia,
            })
            continue

        # 3) Movimento efetivo: aceitamos o melhor vizinho admissível MESMO que
        #    ele seja pior que a solução atual. É esta linha que dá à busca tabu
        #    o poder de escapar de ótimos locais.
        rota_atual = melhor_vizinho
        distancia_atual = melhor_vizinho_distancia

        # 4) O movimento fica proibido pelas próximas `tamanho_tabu` iterações,
        #    impedindo que a troca recém-feita seja desfeita em seguida.
        lista_tabu.append(melhor_movimento)

        # 5) Atualizar a melhor solução global, se houve melhora.
        if distancia_atual < melhor_distancia:
            melhor_rota = rota_atual.copy()
            melhor_distancia = distancia_atual

        historico.append({
            "iteracao": it,
            "rota": rota_atual.copy(),
            "distancia": distancia_atual,
            "melhor_distancia": melhor_distancia,
        })

    return melhor_rota, melhor_distancia, historico
