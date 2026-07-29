"""
main.py
-------

Script principal (execução em linha de comando).

Fluxo:
    1. Gera uma instância do TSP (cidades aleatórias com seed fixa).
    2. Pré-calcula a matriz de distâncias.
    3. Roda a Busca Tabu com parâmetros padrão razoáveis.
    4. Imprime distância inicial x melhor distância encontrada.
    5. Gera os arquivos visuais: rota_final.png, convergencia.png, evolucao.gif.

Uso:
    python main.py
    python main.py --cidades 40 --iteracoes 500 --tabu 20 --seed 7
"""

import argparse
import time

from tsp import busca_tabu, gerar_cidades, matriz_distancias
from visualizacao import gerar_gif, plot_convergencia, plot_rota

# Parâmetros padrão do experimento (os mesmos citados no README).
N_CIDADES = 25
ITERACOES = 300
TAMANHO_TABU = 15
SEED = 42
PASSO_GIF = 5  # 1 frame a cada 5 iterações


def parse_argumentos():
    """Permite variar os parâmetros do experimento sem editar o código."""
    parser = argparse.ArgumentParser(
        description="Busca Tabu aplicada ao Problema do Caixeiro Viajante (TSP)."
    )
    parser.add_argument("--cidades", type=int, default=N_CIDADES,
                        help=f"número de cidades (padrão: {N_CIDADES})")
    parser.add_argument("--iteracoes", type=int, default=ITERACOES,
                        help=f"número de iterações (padrão: {ITERACOES})")
    parser.add_argument("--tabu", type=int, default=TAMANHO_TABU,
                        help=f"tamanho da lista tabu (padrão: {TAMANHO_TABU})")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"semente aleatória (padrão: {SEED})")
    parser.add_argument("--passo-gif", type=int, default=PASSO_GIF,
                        help=f"iterações por frame do GIF (padrão: {PASSO_GIF})")
    parser.add_argument("--sem-gif", action="store_true",
                        help="não gerar o GIF (execução mais rápida)")
    return parser.parse_args()


def main():
    args = parse_argumentos()

    print("=" * 62)
    print(" TSP com Busca Tabu — Otimização Combinatória (IFPI Picos)")
    print("=" * 62)
    print(f"Cidades............: {args.cidades}")
    print(f"Iterações..........: {args.iteracoes}")
    print(f"Tamanho lista tabu.: {args.tabu}")
    print(f"Seed...............: {args.seed}")
    print("-" * 62)

    # --- 1 e 2. Instância do problema -------------------------------------
    cidades = gerar_cidades(args.cidades, seed=args.seed)
    dist = matriz_distancias(cidades)

    # --- 3. Execução da metaheurística ------------------------------------
    inicio = time.perf_counter()
    melhor_rota, melhor_distancia, historico = busca_tabu(
        cidades,
        dist,
        iteracoes=args.iteracoes,
        tamanho_tabu=args.tabu,
        seed=args.seed,
    )
    tempo = time.perf_counter() - inicio

    # O primeiro registro do histórico (iteração 0) é a solução inicial.
    rota_inicial_gerada = historico[0]["rota"]
    distancia_inicial = historico[0]["distancia"]
    melhoria = 100.0 * (distancia_inicial - melhor_distancia) / distancia_inicial

    # --- 4. Resultados numéricos ------------------------------------------
    print(f"Distância inicial (rota aleatória).: {distancia_inicial:.2f}")
    print(f"Melhor distância encontrada........: {melhor_distancia:.2f}")
    print(f"Melhoria...........................: {melhoria:.2f}%")
    print(f"Tempo de execução..................: {tempo:.2f} s")
    print(f"Melhor rota........................: {melhor_rota}")
    print("-" * 62)

    # --- 5. Saídas visuais -------------------------------------------------
    plot_rota(cidades, rota_inicial_gerada, distancia_inicial,
              titulo="Rota inicial (aleatória)",
              caminho_arquivo="rota_inicial.png")
    print("Gerado: rota_inicial.png")

    plot_rota(cidades, melhor_rota, melhor_distancia,
              titulo="Melhor rota encontrada (Busca Tabu)",
              caminho_arquivo="rota_final.png")
    print("Gerado: rota_final.png")

    plot_convergencia(historico, caminho_arquivo="convergencia.png")
    print("Gerado: convergencia.png")

    if not args.sem_gif:
        gerar_gif(cidades, historico, caminho_gif="evolucao.gif",
                  passo=args.passo_gif)
        print("Gerado: evolucao.gif")

    print("=" * 62)
    print("Dica: para explorar a busca interativamente, rode:")
    print("      streamlit run interface.py")
    print("=" * 62)


if __name__ == "__main__":
    main()
