import numpy as np
import matplotlib.pyplot as plt

class Portfolio:
    """
    Description:
        Class to manage funds, changes and relaizations on daily granularity
    """
    def __init__(self, starting_cap, leverage, length):
        self.starting_cap = starting_cap                                    # The starting value of the portfolio  
        self.sentiment = np.zeros(length)                                   # The positions that we take at time [-1, 1]
        self.thetas = np.zeros(length)                                      # The portfolio
        self.thetaPrime = np.zeros(length)                                  # The amount that is sat increasing at the risk free
        self.value = np.zeros(length)                                       # The portfolio value for each day
        self.currentDayIndex = 0                                            # Day index that
        self.leverage = leverage
        self.starting_cap = starting_cap
        self.starting_leveraged_cap = self.leverage * self.starting_cap
        self.dailyReturns = np.zeros(length)                                # Daily actual returns from the stock
    
    def processDay(self, returns, nextDayPredictedReturns, riskFreeRate, standardDeviation):
        """
        Description:
            Process a new day and update positon in the market
        """
        
        # Take a stance on the first day
        if self.currentDayIndex == 0:
            self.value[self.currentDayIndex] = self.starting_leveraged_cap
            if nextDayPredictedReturns > 0:
                self.thetas[self.currentDayIndex] = self.value[self.currentDayIndex]
            else:
                self.thetaPrime[self.currentDayIndex] = self.value[self.currentDayIndex]
        
        # All other day
        else:
            # Start by processing returns going into the day
            self.thetas[self.currentDayIndex] = self.thetas[self.currentDayIndex - 1] * (1 + returns)
            self.thetaPrime[self.currentDayIndex] = self.thetaPrime[self.currentDayIndex - 1] * (1 + riskFreeRate)
            
            self.value[self.currentDayIndex] = self.totalCapitalOnDay(self.currentDayIndex)
            
            # Now handle position adjustment based on the models output
            
        # Start by processing realization over coming into today
        
        # Update daily value and
        self.value[self.currentDayIndex] = self.totalCapitalOnDay(self.currentDayIndex)
        self.currentDayIndex += 1
        
    def totalCapitalOnDay(self, dayIndex):
        return self.thetas[dayIndex] + self.thetaPrime[dayIndex]
    
    def plot(self):
                
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        ax1.plot(self.value)
        ax1.set_title('Portfolio Total Value')
        ax1.set_xlabel('Day')
        ax1.set_ylabel('Value')
        
        ax2.plot(self.thetas)
        ax2.set_title('Thetas (Stock Position)')
        ax2.set_xlabel('Day')
        ax2.set_ylabel('Value')
        
        ax3.plot(self.thetaPrime)
        ax3.set_title('Theta Prime (Risk-Free Position)')
        ax3.set_xlabel('Day')
        ax3.set_ylabel('Value')
        
        plt.tight_layout()
        plt.show()
        
      