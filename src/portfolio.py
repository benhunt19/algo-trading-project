import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

class Portfolio:
    """
    Description:
        Class to manage funds, changes and relaizations on daily granularity
    
    Parameters:
        starting_cap (float): Starting capital of the portfolio (before leverage)
        leverage (float): The leverage amount to adjust the portfolio by
        length (int): How long the portfolio will be run over
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
        self.predictedReturns = np.zeros(length)                            # Returns
        self.maxPredictedReturn = 0                                         # The max predicted return yet
        self.stockData = np.zeros(length)                                   # Final stock data for plotting
        self.graphs = [
            {'title': 'Portfolio Total Value', 'data': self.value},
            {'title': 'Thetas (Stock Position)', 'data': self.thetas},
            {'title': 'Theta Prime (Risk-Free Position)', 'data': self.thetaPrime}
        ]
    
    def processDay(self, returns, nextDayPredictedReturns, riskFreeRate, standardDeviation, threshold=0.3, verbose=True):
        """
        Description:
            Process a new day and update positon in the market

        Parameters:
            returns (float): The percentage change on the stock, going into the day
            nextDayPredictedReturns (float): A models next day prediction for the returns
            riskFreeRate (float): Daily risk free percentage
            standardDeviation (float): Standard deviation of stock in run up
            threshold (float): The threshold to decide the BUY/SELL/NEUTRAL decision
            verbose (bool): Print status of each decision
        """
        
        # Take a stance on the first day
        if self.currentDayIndex == 0:
            
            self.value[self.currentDayIndex] = self.starting_leveraged_cap
            self.thetas[self.currentDayIndex] = self.value[self.currentDayIndex] * 0.5
            self.thetaPrime[self.currentDayIndex] = self.value[self.currentDayIndex] * 0.5
        
        # All other day
        else:
            
            # Start by processing returns going into the day
            if self.thetas[self.currentDayIndex - 1] > 0:
                self.thetas[self.currentDayIndex] = self.thetas[self.currentDayIndex - 1] * (1 + returns - riskFreeRate)
            else:
                self.thetas[self.currentDayIndex] = self.thetas[self.currentDayIndex - 1] * (1 - returns - riskFreeRate)
            
            
            self.thetaPrime[self.currentDayIndex] = self.thetaPrime[self.currentDayIndex - 1] #* (1 + riskFreeRate)
            self.value[self.currentDayIndex] = self.totalCapitalOnDay(self.currentDayIndex)
            self.predictedReturns[self.currentDayIndex] = nextDayPredictedReturns
            
            # Maintain data for the maximum predicted returns
            if abs(nextDayPredictedReturns) > self.maxPredictedReturn:
                self.maxPredictedReturn = abs(nextDayPredictedReturns)
            
            # Now handle position adjustment based on the models output
                        
            if nextDayPredictedReturns > threshold:
                if verbose:
                    print('BUY - GOING LONG')
                # Process increase in theta
                if self.thetas[self.currentDayIndex] < self.totalCapitalOnDay():
                    
                    # The strength of the position needs to be determined by the strength of the signal
                    # buyStrength = 0.7
                    buyStrength = min(nextDayPredictedReturns / self.maxPredictedReturn, 1)
                    dayCapStart = self.totalCapitalOnDay()
                    
                    self.thetas[self.currentDayIndex] = buyStrength * dayCapStart
                    self.thetaPrime[self.currentDayIndex] = dayCapStart - abs(self.thetas[self.currentDayIndex])
                    
                # if at the threshold, then stay where it is    
                else:
                    pass
                
            elif abs(nextDayPredictedReturns) <= threshold:
                
                if verbose:
                    print("NEUTRAL - UPDATING TO HOLD RISK FREE")
                
                # Chuck all in the risk free for the timebeing
                self.thetaPrime[self.currentDayIndex] = self.totalCapitalOnDay()
                self.thetas[self.currentDayIndex] = 0
            
            elif nextDayPredictedReturns < -threshold:
                
                if verbose:
                    print("SELL - GOING SHORT")
                
                if abs(self.thetas[self.currentDayIndex]) < self.totalCapitalOnDay():
                    
                    # The strength of the position needs to be determined by the strength of the signal
                    # sellStrength = 1
                    
                    sellStrength = min(abs(nextDayPredictedReturns) / self.maxPredictedReturn, 1)
                    dayCapStart = self.totalCapitalOnDay()
                    
                    dayCapStart = self.totalCapitalOnDay()
                    
                    self.thetas[self.currentDayIndex] = sellStrength * -dayCapStart
                    self.thetaPrime[self.currentDayIndex] = dayCapStart - abs(self.thetas[self.currentDayIndex])
                    
                # if at the threshold, then stay where it is    
                else:
                    pass
            
        # Update daily value and finish eith updating the day index
        if verbose:
            self.printDayUpdate()
            
        self.value[self.currentDayIndex] = self.totalCapitalOnDay()
        self.currentDayIndex += 1
        
    def totalCapitalOnDay(self, dayIndex=None):
        dayIndex = self.currentDayIndex if dayIndex is None else dayIndex
        return abs(self.thetas[dayIndex]) + self.thetaPrime[dayIndex]
    
    def printDayUpdate(self, dayIndex=None):
        dayIndex = self.currentDayIndex if dayIndex is None else dayIndex
        update = f'Theta: {self.thetas[dayIndex]}, ThetaPrime: {self.thetaPrime[dayIndex]}, Total: {self.totalCapitalOnDay(dayIndex=dayIndex)}'
        print(update)
    
    
    def plot(self):
        
        sns.set_style("darkgrid")
                
        if self.stockData is not None:
            self.graphs.append({'title': 'Stock Price', 'data': self.stockData},)
                
        fig, axis = plt.subplots(2, 2, figsize=(10, 3 * len(self.graphs)))
        plotGap = 1.5
        
        # Flatten the 2x2 axis array for easier iteration
        axis_flat = axis.flatten()
        for i, ax in enumerate(axis_flat):
            if i >= len(self.graphs):
                break
            
            ax.plot(self.graphs[i]['data'])
            ax.set_title(self.graphs[i]['title'])
            ax.set_xlabel('Day')
            ax.set_ylabel('Value')
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=20))
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=30))
            ax.margins(y=plotGap)
            
        plt.tight_layout()
        plt.show()
    
    def startLivePlot(self):
        plt.ion()
        sns.set_style("darkgrid")
        
        if self.stockData is not None:
            self.graphs.append({'title': 'Stock Price', 'data': self.stockData})
        
        fig, axis = plt.subplots(2, 2, figsize=(10, 3 * len(self.graphs)))
        axis_flat = axis.flatten()
        plotGap = 1.5
        
        self.lines = [None for _ in self.graphs]
        
        for i, ax in enumerate(axis_flat):
            if i >= len(self.graphs):
                break
    
            # Initialize empty line with just the first point
            self.lines[i], = ax.plot(self.graphs[i]['data'][:1])
            ax.set_title(self.graphs[i]['title'])
            ax.set_xlabel('Day')
            ax.set_ylabel('Value')
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=20))
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=30))
            ax.margins(y=plotGap)
        
        plt.tight_layout()
        plt.show()
    
        def updatePlot():
            for i, line in enumerate(self.lines):
                if line is not None:
                    # Update both x and y data up to current day
                    line.set_xdata(range(self.currentDayIndex))
                    line.set_ydata(self.graphs[i]['data'][:self.currentDayIndex])
                    ax = line.axes
                    ax.relim()
                    ax.autoscale_view()
            
            plt.draw()
            plt.pause(0.001)
    
        self.updatePlot = updatePlot

    def updatePlot(self):
        # Update all self.graphs
        for i, line in enumerate(self.lines):
            if line is not None:
                line.set_ydata(self.graphs[i]['data'][:self.currentDayIndex])
                ax = line.axes
                ax.relim()
                ax.autoscale_view()

        plt.draw()  # Redraw the plot
        plt.pause(0.01)  # Pause to allow the plot to update
        
    def sharpeRatio(self):
        finalReturns = self.value[-1] - self.starting_leveraged_cap
        return finalReturns / self.value.std()