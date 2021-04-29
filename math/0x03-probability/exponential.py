#!/usr/bin/env python3
"""
Represents an exponential distribution
"""


e = 2.7182818285


class Exponential:
    """Represents an exponential distribution"""
    def __init__(self, data=None, lambtha=1.):
        """   Class constructor   """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                # save lambtha as a float
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                # calculate lambtha of the data
                self.lambtha = float(1/(sum(data)/len(data)))

    def pdf(self, x):
        """
        Calculates PDF for a given time period x is the time period
        """
        if x < 0:
            return 0
        return self.lambtha * e ** ((-1) * self.lambtha * x)

    def cdf(self, x):
        """
        CalculatesCDF for a given time period
        """
        if x < 0:
            return 0
        return 1 - e ** ((-1) * self.lambtha * x)
