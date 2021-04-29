#!/usr/bin/env python3
""" Represents a binomial distribution """


class Binomial:
    """ Represents a binomial distribution """

    def __init__(self, data=None, n=1, p=0.5):
        """ Constructor of the class  """
        if data is not None:
            if type(data) != list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = 0.
            counter = 0
            for element in data:
                if type(element) not in {int, float}:
                    raise TypeError('Each element in data must be a number')
                counter += 1
                mean += element
            mean /= counter
            varian = 0.
            for element in data:
                varian += ((mean - element) ** 2)
            varian /= counter
            p = 1 - varian / mean
            self.n = round(mean / p)
            self.p = float(mean / self.n)
        else:
            if type(n) not in {int, float} or n <= 0:
                raise ValueError('n must be a positive value')
            self.n = round(n)
            if not 0 < p < 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.p = float(p)

    def pmf(self, k):
        """
        Calculates PMF for a given number  .
        """
        if type(k) not in {int, float}:
            raise TypeError('k must be a number')
        k = int(k)
        if k < 0:
            return 0
        n = self.n
        p = self.p
        comb = Binomial.combinations
        return comb(n, k) * p ** k * (1 - p) ** (n - k)

    def cdf(self, k):
        """
        Calculates  CDF for a given number
        """
        if type(k) not in {int, float}:
            raise TypeError('k must be a number')
        k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf

    @staticmethod
    def combinations(n, r):
        """
        Calculates the number of combinations.
        """
        fact = Binomial.fact
        return fact(n) / (fact(r) * fact(n - r))

    @staticmethod
    def fact(n):
        """ Calculates the factorial of a number. """
        if type(n) != int or n < 0:
            raise ValueError('n must be a positive integer or 0.')
        answer = 1
        for i in range(2, n + 1):
            answer *= i
        return answer
