import os
import time
import numpy as np
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



# Plot training hisory
def plot_history(train_loss, train_dice,val_loss, val_dice, save_path):
  
    # Colors (we assume there are no more than 7 metrics):
    colors = ['r', 'g', 'k', 'm', 'c', 'y', 'w']

    # Find the best epoch
    best_train_loss = np.min(train_loss)
    best_train_dice = np.max(train_dice)
    best_val_loss = np.min(val_loss)
    best_val_dice = np.max(val_dice)

    
    # Initialize figure:
    # Axis 1 will be for metrics, and axis 2 for losses.
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    # Plotting
    ax2.plot(train_loss, 'b-', label='{} ({:.3f})'.format('Train Loss', best_train_loss))
    ax1.plot(train_dice, 'r-', label='{} ({:.3f})'.format('Train IoU', best_train_dice))
    ax2.plot(val_loss, 'b--', label='{} ({:.3f})'.format('Val Loss', best_val_loss))
    ax1.plot(val_dice, 'r--', label='{} ({:.3f})'.format('Val IoU', best_val_dice))

    ax1.set_ylim(0,1)
    ax2.set_ylim(0,0.5)
    
    # Add title
    plt.title('Model training history')

    # Add axis labels
    ax1.set_ylabel('Metric')
    ax2.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    
    # ??
    fig.tight_layout()

    # Add legend
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Save fig
    plt.savefig(save_path)

    # Close plot
    plt.close()
    
def plot_history_offline(train_dices, val_dices, extra_val_dices, train_hds, val_hds, extra_val_hds, save_path):


    # Find the best epoch
    best_train_dice = np.max(train_dices)
    best_train_hd = np.min(train_hds)
    
    best_val_dice = np.max(val_dices)
    best_val_hd = np.min(val_hds)

    best_extra_val_dice = np.max(extra_val_dices)
    best_extra_val_hd = np.min(extra_val_hds)
    
    # Initialize figure:
    # Axis 1 will be for metrics, and axis 2 for losses.
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    # Plotting
    ax1.plot(train_dices, 'r', label='{} ({:.3f})'.format('Train Dice', best_train_dice))
    ax2.plot(train_hds, 'r--', label='{} ({:.3f})'.format('Train HD', best_train_hd))
    
    ax1.plot(val_dices, 'b', label='{} ({:.3f})'.format('Val Dice', best_val_dice))
    ax2.plot(val_hds, 'b--', label='{} ({:.3f})'.format('Val HD', best_val_hd))

    ax1.plot(extra_val_dices, 'g', label='{} ({:.3f})'.format('Extra Val Dice', best_extra_val_dice))
    ax2.plot(extra_val_hds, 'g--', label='{} ({:.3f})'.format('Extra Val HD', best_extra_val_hd))

    ax1.set_ylim(0,1)
    ax2.set_ylim(0,60)
    
    # Add title
    plt.title('Model training history')

    # Add axis labels
    ax1.set_ylabel('Dice')
    ax2.set_ylabel('HD (mm)')
    ax1.set_xlabel('Epoch')
    
    # ??
    fig.tight_layout()

    # Add legend
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Save fig
    plt.savefig(save_path)

    # Close plot
    plt.close()