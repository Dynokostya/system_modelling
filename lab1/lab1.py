import numpy as np
import random
import matplotlib.pyplot as plt


class RandomGenerator:
    def __init__(self, m, z, mod):
        self.m = m
        self.z = z
        self.mod = mod

    def next_float(self):
        self.z = self.m * self.z % self.mod
        return self.z / self.mod

    def next_int(self):
        self.z = self.m * self.z % self.mod
        return self.z

    def beta_dist(self, alpha, beta):
        u1 = self.next_float()
        u2 = self.next_float()

        x = u1 ** 1 / alpha
        y = u2 ** 1 / beta
        return x / (x + y)

    def beta_dist_neumann_pearson(self, alpha, beta, length):
        def beta_pdf(x):
            if 0 <= x <= 1:
                return (x**(alpha-1)) * ((1-x)**(beta-1))
            return 0
        res = []
        for _ in range(length):
            while True:
                u1 = self.next_float()
                u2 = self.next_float()
                fx = beta_pdf(u1)
                if u2 <= fx:
                    res.append(u1)
                    break
        return res


def task1(generator, length):
    print('---Task 1---')
    nums = [generator.next_float() for _ in range(length)]
    print(*nums, sep=', ')
    print()


def task2(generator, length):
    print('---Task 2---')
    nums = [generator.next_int() for _ in range(length)]
    print(*nums, sep=', ')
    print()


def task3(generator, length):
    print('---Task 3---')
    nums = [generator.beta_dist(3, 1) for _ in range(length)]
    print(*nums, sep=', ')
    print()
    plt.figure(figsize=(12, 6))
    plt.title('Beta Distribution')
    plt.scatter(nums[:length//2], nums[length//2:])
    plt.show()


def task4(generator, length):
    print('---Task 4---')
    nums = generator.beta_dist_neumann_pearson(3, 1, length)
    print(*nums, sep=', ')
    print()


def task5_6(generator: RandomGenerator, length):
    def empirical_cdf(data):
        n = len(data)
        x = sorted(data)
        y = np.arange(1, n+1) / n
        return x, y

    def plot_graph(nums, norm, min, max):
        plt.clf()
        # Графік емпіричної функції розподілу
        x, y = empirical_cdf(nums)
        plt.plot(x, y, label='EDF (Random)')
        # Графік теоретичної функції розподілу
        x = np.linspace(min, max, length)
        plt.scatter(x, norm, marker='o', color='black', label='Theoretical CDF (Random)', alpha=0.5)
        plt.legend()
        plt.title('Empirical and Theoretical CDFs')
        plt.xlabel('Value')
        plt.ylabel('Probability')
        plt.show()

    nums0 = [generator.beta_dist(3, 1) for _ in range(length)]
    nums0_norm = [random.betavariate(3, 1) for _ in range(length)]
    plot_graph(nums0, nums0_norm, 0, 1)

    nums1 = [random.random() for _ in range(length)]
    nums1_norm = [random.random() for _ in range(length)]
    plot_graph(nums1, nums1_norm, 0, 1)

    nums2 = [random.randint(1, 2**32) for _ in range(length)]
    nums2_norm = [random.uniform(1, 2**32) for _ in range(length)]
    plot_graph(nums2, nums2_norm, 1, 2**32)

    nums3 = [random.betavariate(3, 1) for _ in range(length)]
    nums3_norm = [random.betavariate(3, 1) for _ in range(length)]
    plot_graph(nums3, nums3_norm, 0, 1)


def task7():
    print('---Task 7---')
    all_combinations = [[(dice1, dice2) for dice1 in range(1, 7)] for dice2 in range(1, 7)]
    print(all_combinations)
    print()


if __name__ == '__main__':
    M = random.randint(1, 2**32)
    Z = 123456789
    MOD = 2**32
    num_amount = 100
    rand = RandomGenerator(M, Z, MOD)
    rand2 = RandomGenerator(477211307, 123456789, MOD)
    task1(rand, num_amount)
    task2(rand2, num_amount)
    task3(rand, num_amount)
    task4(rand, num_amount)
    task5_6(rand, num_amount)
    task7()
