import pygame as pg 
import numpy as np 
import pandas as pd 


def draw_grid(screen, render_size, candle_start_height, nb_y_label, alpha_screen, scaling_data, font): 

    alpha_screen.fill([0,0,0,0])

    x_label_pos = render_size[0] * 0.02
    y_label_inc = render_size[1] * (1. - candle_start_height) / nb_y_label
    y_label_pos = render_size[1] * (1. - candle_start_height)
    y_labels = np.linspace(scaling_data.min(), scaling_data.max(), nb_y_label)
    x_labels = np.linspace(0, render_size[1], nb_y_label)
    for i in range(nb_y_label): 
        # DRAWING GRID 
        pg.draw.line(alpha_screen, (220,220,220,50), 
            np.array([0, y_label_pos - 5]).astype(int), np.array([render_size[0], y_label_pos - 5]).astype(int) )
        pg.draw.line(alpha_screen, (220,220,220,50), 
            np.array([x_labels[i], 0]).astype(int), np.array([x_labels[i], render_size[1]]).astype(int) )

        # WRITING LABELS
        label = font.render('{:.1f}'.format(y_labels[i]), 50, (250,250,250))
        screen.blit(label, np.array([x_label_pos, y_label_pos - 30]).astype(int))
        y_label_pos -= y_label_inc


def draw_candles(screen, render_size, candle_start_height, alpha_screen,data, scaling_data, y_magn, colors): 

    rect_width = 2 * (render_size[0]/data.shape[0])

    x_pos = render_size[0]* 0.5 - rect_width * data.shape[0] * 0.5
    # y_pos = data[:,1] + data[:,2]
    y_pos = data.High + data.Low
    
    x_pos_vol = [x_pos]

    
    color = colors[0]
    for i in range(data.shape[0]): 

        rect_height = np.abs(data.Open[i] - data.Close[i]) * y_magn
        rect_center = ((data.Open[i] + data.Close[i]) * 0.5 - scaling_data.min()) * y_magn 

        shape = np.array([x_pos, 
                         render_size[1] * (1. - candle_start_height) - rect_center, 
                         rect_width * 0.9,
                         0.5 * rect_height], dtype = int)
        

        line_height = np.abs(data.High[i] - data.Low[i]) * y_magn
     
        line_up = np.array([x_pos + rect_width * 0.5 - 2, render_size[1] * (1. - candle_start_height) -  rect_center + 0.5 * line_height]).astype(int)
        line_down = np.array([x_pos + rect_width * 0.5 - 2, render_size[1] * (1. - candle_start_height) -  rect_center - 0.5 * line_height]).astype(int)
        
        line_height = line_height.astype(int)
        
        if i > 0: 
            color = colors[1] if (data.Open[i] + data.High[i]) * 0.5 > (data.Open[i-1] + data.High[i-1])*0.5 else colors[0] 

        pg.draw.rect(screen, color, list(shape))
        pg.draw.line(alpha_screen, (250,250,250,80), line_up, line_down, width = 2)

        x_pos += rect_width
        x_pos_vol.append(x_pos)

    return rect_width, x_pos_vol

def draw_agent(screen, render_size, 
              candle_start_height, alpha_screen,
              scaling_data, y_magn, orders_hist, orders_color,
              rect_width, colors, font, offset = 40): 

    center_pos_x = render_size[0] * 0.5 
    screen.blit(alpha_screen,(0,0))

    orders_length = len(list(orders_hist.values())[0].net_worth_history)

    for i in reversed(range(1, orders_length)):
        current_heights = []

        for trader, trader_color in zip(list(orders_hist.values()), list(orders_color.values())):  

            order_hist = trader.net_worth_history
            height_current = render_size[1] * (1. - candle_start_height) - (order_hist[i] - scaling_data.min()) * y_magn
            height_previous = render_size[1] * (1. - candle_start_height) - (order_hist[i-1] - scaling_data.min()) * y_magn


            pg.draw.line(screen, trader_color, 
                         np.array([center_pos_x, height_current]).astype(int), 
                         np.array([center_pos_x - rect_width, height_previous]).astype(int), width = 8)

            current_heights.append(height_current)


        if i == orders_length -1 :
            for j in range(1, len(current_heights)):
                line_col = orders_color[list(orders_color.keys())[j]] if current_heights[j] > current_heights[0] else (0,0,0) 
                pg.draw.line(screen, line_col, 
                        np.array([center_pos_x + offset*(j-1), current_heights[j]]).astype(int), 
                        np.array([center_pos_x + offset*(j-1), current_heights[j-1]]).astype(int), width = 2)
                for h in [current_heights[j], current_heights[j-1]]: 
                    pg.draw.line(screen, line_col, 
                        np.array([center_pos_x - 5, h - 5]).astype(int) + np.array([offset*(j-1),0]).astype(int), 
                        np.array([center_pos_x + 5, h + 5]).astype(int) + np.array([offset*(j-1),0]).astype(int), width = 2)
                

                info = font.render('{:.1f}'.format(orders_hist[list(orders_hist.keys())[0]].net_worth_history[-1] - orders_hist[list(orders_hist.keys())[j]].net_worth_history[-1]), 
                                                    50, line_col)
                screen.blit(info, np.array([center_pos_x + j * offset + 20, 0.5 * (current_heights[j] + current_heights[j-1])]).astype(int))

        # if show_random_trader: 
        #     random_trader_height_current = render_size[1] * (1. - candle_start_height) - (random_order_hist[i] - scaling_data.min()) * y_magn
        #     random_trader_height_previous = render_size[1] * (1. - candle_start_height) - (random_order_hist[i-1] - scaling_data.min()) * y_magn
        #     pg.draw.line(screen, colors[0], 
        #              np.array([center_pos_x, random_trader_height_current]).astype(int), 
        #              np.array([center_pos_x - rect_width, random_trader_height_previous]).astype(int), width = 2)
        #     if i == len(order_hist) -1:

        #         line_col = [239, 192, 0] if height_current < random_trader_height_current else (0,0,0) 
        #         pg.draw.line(screen, line_col, 
        #                 np.array([center_pos_x, random_trader_height_current]).astype(int), 
        #                 np.array([center_pos_x, height_current]).astype(int))
        #         for h in [height_current, random_trader_height_current]: 
        #             pg.draw.line(screen, line_col, 
        #                 np.array([center_pos_x - 5, h - 5]).astype(int), 
        #                 np.array([center_pos_x + 5, h + 5]).astype(int))
        #         info = font.render('{:.1f}'.format(order_hist[-1] - random_order_hist[-1]), 50, line_col)
        #         screen.blit(info, np.array([center_pos_x + 20, 0.5 * (height_current + random_trader_height_current)]).astype(int))


        center_pos_x -= rect_width


def draw_volumes(screen, render_size, graph_height_ratio, alpha_screen, volumes, volume_magn, x_pos_vol): 
    volumes = np.hstack([np.array(x_pos_vol[1:]).reshape(-1,1), volumes.reshape(-1,1)])
    volumes[:,1] = render_size[1] - volumes[:,1] * volume_magn
    volumes = np.vstack([np.array([volumes[0,0], render_size[1]]).reshape(1,-1), 
                         volumes,
                         np.array([volumes[-1,0], render_size[1]]).reshape(1,-1)])
    pg.draw.line(screen, (0,0,0), np.array([0, render_size[1] *  graph_height_ratio - 5]).astype(int), np.array([render_size[0], render_size[1] * graph_height_ratio - 5]).astype(int), width = 6)
    pg.draw.polygon(screen, ((150,30,30)), volumes.astype(int), width = 0)

def draw_infos(env): 
    viz_data = "Steps:                       {}/{}\nNet_Worth:            {:.2f}\nBaseline:                 {:.2f}\nBaselineDelta:            {:.2f}\nEp_Reward:             {:.2f}\nReward:                   {:.2f}\nBalance:                 {:.2f}\nHeld:                        {:.3f}\nBought:                  {:.3f}\nSold:                        {:.3f}\nOrders:                     {}".format(env.current_ts,
                                                                                                                                   env.ep_timesteps,
                                                                                                                                   env.trader.net_worth, 
                                                                                                                                   env.baseline_trader.net_worth,
                                                                                                                                   env.trader.net_worth - env.baseline_trader.net_worth,  
                                                                                                                                   env.ep_reward,
                                                                                                                                   env.compute_reward(), 
                                                                                                                                   env.trader.balance,
                                                                                                                                   env.trader.stock_held,
                                                                                                                                   env.trader.stock_bought, 
                                                                                                                                   env.trader.stock_sold, 
                                                                                                                                   env.nb_ep_orders)
    for i,vz in enumerate(viz_data.split('\n')): 
        label = env.font.render(vz, 50, (250,250,250))
        env.screen.blit(label, np.array([env.render_size[0] * 0.7, env.render_size[1] * (0.45 + i * 0.05)]).astype(int))