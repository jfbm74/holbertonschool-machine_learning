#!/usr/bin/env python3
""" This module contains the Normal class. """


class Normal:
    """ Class that represents a Normal distribution. """

    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Constructor of the class
        """
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
            self.mean = mean / counter
            mean = 0
            counter = 0
            for element in data:
                counter += 1
                mean += (self.mean - element) ** 2
            self.stddev = (mean / counter) ** 0.5
        else:
            if type(stddev) not in {int, float} or stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.mean = float(mean)
            self.stddev = float(stddev)

    def pdf(self, x):
        """
        Calculates PDF for a given
        """
        number = (Normal.e ** (-1 / 2 * ((x - self.mean) / self.stddev) ** 2))
        dens = (self.stddev * (2 * Normal.pi) ** 0.5)
        return number / dens

    def cdf(self, x):
        """
        Calculates CDF for a given value .
        """
        return (1 + Normal.erf((x - self.mean) / (self.stddev * 2 ** 0.5))) / 2

    def z_score(self, x):
        """
        Calculates  z-score of a given value.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates  value of  score.
        """
        return z * self.stddev + self.mean

    @staticmethod
    def erf(x):
        return 2 / Normal.pi ** 0.5 * (x - x ** 3 / 3 + x ** 5 / 10 -
                                       x ** 7 / 42 + x ** 9 / 216)
