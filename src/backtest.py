from Stock import *
import signals
import pandas as pd
import plotter
from strategy import AdvancedStrategy


def backtest(stock,strategy='advanced',plot=False) :
    """
    Performs a backtest of a given stock using the specified trading strategy.

    Args:
        stock (Stock): The stock object to backtest.
        strategy (str, optional): The trading strategy to use for the backtest. Defaults to 'hybrid'.
        plot (bool, optional): Whether to plot the results of the backtest. Defaults to False.

    Returns:
        None
    """

    df = pd.read_csv(stock.file)
    pd.to_datetime(df['date'])

    pos = 0
    num = 0
    percentchange = []
    buy_dates = []
    sell_dates = []

    for i in df.index:
        close = df['close'][i]
        date = df['date'][i]

        df_trimmed = df.copy(deep=True)
        df_trimmed= df_trimmed.loc[df_trimmed['date'] <= str(date)]

        if strategy == 'advanced':
            signal = AdvancedStrategy(stock.symbol).generate_signal(df_trimmed)[0]
        
        else:
            strat = getattr(signals,strategy)
            signal = strat(df_trimmed)
        

        if signal > 0 and pos==0:
            bp=close
            pos=1
            buy_dates.append(date)
        
        elif signal <0  and pos==1:
            pos=0
            sp=close
            pc=(sp/bp-1)*100
            percentchange.append(pc)
            sell_dates.append(date)
    
    #If its last day, sell the units
        if(num==df["close"].count()-1 and pos==1):
            pos=0
            sp=close
            pc=(sp/bp-1)*100
            percentchange.append(pc)

        num+=1

    print(percentchange)

    #calculate gains
    gains=0
    ng=0
    losses=0
    nl=0
    totalR=1

    for i in percentchange:
        if(i>0):
            gains+=i
            ng+=1
        else:
            losses+=i
            nl+=1
        totalR=totalR*((i/100)+1)

    totalR=round((totalR-1)*100,2)

    if(ng>0):
        avgGain=gains/ng
        maxR=str(max(percentchange))
    else:
        avgGain=0
        maxR="undefined"

    if(nl>0):
        avgLoss=losses/nl
        maxL=str(min(percentchange))
        ratio=str(-avgGain/avgLoss)
    else:
        avgLoss=0
        maxL="undefined"
        ratio="inf"

    if(ng>0 or nl>0):
        battingAvg=ng/(ng+nl)
    else:
        battingAvg=0

    #print results
    print()
    print("Results for "+ stock() +" going back to "+str(df.date[0])+", Sample size: "+str(ng+nl)+" trades"+" using " + strategy + " strategy.")
    print("Batting Avg: "+ str(battingAvg))
    print("Gain/loss ratio: "+ ratio)
    print("Average Gain: "+ str(avgGain))
    print("Average Loss: "+ str(avgLoss))
    print("Max Return: "+ maxR)
    print("Max Loss: "+ maxL)
    print("Total return over "+str(ng+nl)+ " trades: "+ str(totalR)+"%")
    print()

    #plot results
    if plot:
        plotter.plot(df,buy_dates,sell_dates)

if __name__ == '__main__':
    backtest(Stock("akjcl"),plot=True)