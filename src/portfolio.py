import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd

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
        self.thetas = np.zeros(length)                                      # The portfolio
        self.thetaPrime = np.zeros(length)                                  # The amount that is sat increasing at the risk free
        self.value = np.zeros(length)                                       # The portfolio value for each day
        self.currentDayIndex = 0                                            # Day index that
        self.leverage = leverage
        self.starting_cap = starting_cap
        self.PnL = np.zeros(length)                                         # The differences in theta
        self.capitalGains = np.zeros(length)                                # The capital gains from the risk free
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
            self.thetas[self.currentDayIndex] = self.value[self.currentDayIndex] * 0
            self.thetaPrime[self.currentDayIndex] = self.value[self.currentDayIndex] 
        
        # All other day
        else:
            
            # Start by processing returns going into the day, the returns based on the sign and also reducing by the risk free rate as payment for the leverage
            if self.thetas[self.currentDayIndex - 1] > 0:
                gainFromMarket = self.thetas[self.currentDayIndex - 1] * ( returns - riskFreeRate)
                self.thetas[self.currentDayIndex] = self.thetas[self.currentDayIndex - 1] +  gainFromMarket
            else:
                gainFromMarket = abs(self.thetas[self.currentDayIndex - 1]) * (- returns - riskFreeRate)
                self.thetas[self.currentDayIndex] = self.thetas[self.currentDayIndex - 1] - gainFromMarket
                
            self.PnL[self.currentDayIndex] = gainFromMarket
            
            capitalGain = self.thetaPrime[self.currentDayIndex - 1] * max((riskFreeRate / self.leverage), 0)
            self.thetaPrime[self.currentDayIndex] = self.thetaPrime[self.currentDayIndex - 1] + capitalGain
            self.capitalGains[self.currentDayIndex] = capitalGain
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
        """
        Description:
            Get the total capital on a day, this is based on the thetas
            and the risk free capital spare
        """
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
        plotGap = 2
        
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
        plotGap = 2
        
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
    
    
    def returnsOnOriginal(self) -> pd.DataFrame:
        return (self.value[-1] - self.starting_leveraged_cap) / self.starting_cap
    
    def sharpeRatio(self):
        # Calculate daily returns
        daily_returns = np.diff(self.value) / self.value[:-1]
        
        # Calculate the mean and standard deviation of daily returns
        mean_daily_return = np.mean(daily_returns)
        std_daily_return = np.std(daily_returns)
        
        # Annualize the Sharpe Ratio (assuming 252 trading days in a year)
        sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(252)
        
        return sharpe_ratio
    
    def maxDrawdown(self):
        """Calculate the maximum drawdown"""
        running_max = np.maximum.accumulate(self.value)
        drawdowns = (self.value - running_max) / running_max
        return abs(min(drawdowns))
    
    def calmarRatio(self):
        """Calculate the Calmar ratio"""
        # Calculate total and annualized returns
        total_return = (self.value[-1] - self.value[0]) / self.value[0]
        days = len(self.value)
        annualized_return = (1 + total_return) ** (252/days) - 1
        
        # Get max drawdown
        max_drawdown = self.maxDrawdown()
        
        # Calculate Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown != 0 else 0
        
        return calmar_ratio