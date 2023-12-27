import random


def spearman_coefficient(rank1, rank2):
    n = len(rank1)
    d_sum = sum((rank1[i] - rank2[i]) ** 2 for i in range(n))
    coefficient = 1 - (6 * d_sum) / (n ** 3 - n)
    return coefficient


def kendall_coefficient(rank1, rank2):
    n = len(rank1)
    concordant = discordant = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            concordant += (rank1[i] - rank1[j]) * (rank2[i] - rank2[j]) > 0
            discordant += (rank1[i] - rank1[j]) * (rank2[i] - rank2[j]) < 0
    coefficient = (concordant - discordant) / (0.5 * n * (n - 1))
    return coefficient


def generalized_correlation_coefficient(rank1, rank2):
    spearman = spearman_coefficient(rank1, rank2)
    kendall = kendall_coefficient(rank1, rank2)

    n = len(rank1)
    k = 2 * (spearman - kendall)
    generalized_coefficient = (n - 1) * spearman - k * (n - 2) / 2

    return generalized_coefficient


def main():
    n = int(input("Type the size of ranks: "))
    # # Random data
    # rank1 = [random.randint(1, n) for _ in range(n)]
    # rank2 = [random.randint(1, n) for _ in range(n)]
    # Similar data
    rank1 = rank2 = [i for i in range(1, n)]

    spearman = spearman_coefficient(rank1, rank2)
    kendall = kendall_coefficient(rank1, rank2)
    generalized_correlation = generalized_correlation_coefficient(rank1, rank2)

    print("rank1:", rank1)
    print("rank2:", rank2)
    print("Spearman:", spearman)
    print("Kendall:", kendall)
    print("Generalized correlation:", generalized_correlation)


main()
