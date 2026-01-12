# src/analysis.py

"""
Analysis utilities - Calculate performance metrics
"""

import numpy as np


def calculate_stats(equity_curve, trades):
    """Calculate backtest statistics."""
    if not equity_curve:
        return {}
    
    dates, equities = zip(*equity_curve)
    equities = list(equities)
    
    final_equity = equities[-1]
    initial_equity = equities[0]
    total_return = (final_equity - initial_equity) / initial_equity * 100
    
    # Trade stats
    exit_trades = [t for t in trades if t['type'] == 'exit']
    
    if exit_trades:
        pnl_values = [t['realized_pnl'] for t in exit_trades]
        wins = len([p for p in pnl_values if p > 0])
        losses = len([p for p in pnl_values if p < 0])
        win_rate = wins / len(exit_trades) * 100
        
        avg_win = np.mean([p for p in pnl_values if p > 0]) if wins > 0 else 0
        avg_loss = np.mean([p for p in pnl_values if p < 0]) if losses > 0 else 0
        total_pnl = sum(pnl_values)
    else:
        wins = losses = win_rate = 0
        avg_win = avg_loss = total_pnl = 0
    
    # Equity stats
    max_equity = max(equities)
    min_equity = min(equities)
    max_drawdown = (min_equity - initial_equity) / initial_equity * 100
    
    return {
        'final_equity': final_equity,
        'total_return': total_return,
        'max_equity': max_equity,
        'min_equity': min_equity,
        'max_drawdown': max_drawdown,
        'total_trades': len(exit_trades),
        'winning_trades': wins,
        'losing_trades': losses,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'total_pnl': total_pnl,
    }


def print_results(name, stats, initial_capital):
    """Pretty print results."""
    if not stats:
        print(f"\n{name}: No trades\n")
        return
    
    print(f"\n{name}")
    print("-"*70)
    print(f"Final Equity:        ${stats['final_equity']:,.2f}")
    print(f"Total Return:        {stats['total_return']:+.2f}%")
    print(f"Total PnL:           ${stats['total_pnl']:,.2f}")
    print(f"Max Drawdown:        {stats['max_drawdown']:.2f}%")
    print(f"\nTrades:              {stats['total_trades']}")
    print(f"Winning:             {stats['winning_trades']}")
    print(f"Losing:              {stats['losing_trades']}")
    print(f"Win Rate:            {stats['win_rate']:.1f}%")
    print(f"Avg Win:             ${stats['avg_win']:,.2f}")
    print(f"Avg Loss:            ${stats['avg_loss']:,.2f}")
