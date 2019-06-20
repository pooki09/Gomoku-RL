import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from random import randint
import openpyxl
#from google.colab import files

def add_stone(board_matrix, x, y, flag):
    board_matrix[x-1, y-1] = flag
    return board_matrix

def print_board(board_matrix):
    print('| |1|2|3|4|5|6|7|')
    for i in range(7):
        print('|{}'.format(i+1),end='')
        for j in range(7):
            if board_matrix[i,j] == 1:
                print('|o',end='')
            elif board_matrix[i,j] == 2:
                print('|x',end='')
            else:
                print('| ',end='')
            if j == 6:
                print('|')

def isempty(board_matrix,x,y):
    if board_matrix[x-1, y-1] == 0:
        return True
    else:
        return False     

def check_end(board_matrix):
    for i in range(np.shape(board_matrix)[0]):
        for j in range(np.shape(board_matrix)[1]-4):
            if board_matrix[i,j] == 1 and board_matrix[i,j+1] == 1 and board_matrix[i,j+2] == 1 and board_matrix[i,j+3] == 1 and board_matrix[i,j+4] == 1:
                return 1
            elif board_matrix[i,j] == 2 and board_matrix[i,j+1] == 2 and board_matrix[i,j+2] == 2 and board_matrix[i,j+3] == 2 and board_matrix[i,j+4] == 2:
                return 2
    for j in range(np.shape(board_matrix)[1]):
        for i in range(np.shape(board_matrix)[0]-4):
            if board_matrix[i,j] == 1 and board_matrix[i+1,j] == 1 and board_matrix[i+2,j] == 1 and board_matrix[i+3,j] == 1 and board_matrix[i+4,j] == 1:
                return 1
            elif board_matrix[i,j] == 2 and board_matrix[i+1,j] == 2 and board_matrix[i+2,j] == 2 and board_matrix[i+3,j] == 2 and board_matrix[i+4,j] == 2:
                return 2
    for i in range(np.shape(board_matrix)[0]-4):
        for j in range(np.shape(board_matrix)[1]-4):
            if board_matrix[i,j] == 1 and board_matrix[i+1,j+1] == 1 and board_matrix[i+2,j+2] == 1 and board_matrix[i+3,j+3] == 1 and board_matrix[i+4,j+4] == 1:
                return 1
            elif board_matrix[i,j] == 2 and board_matrix[i+1,j+1] == 2 and board_matrix[i+2,j+2] == 2 and board_matrix[i+3,j+3] == 2 and board_matrix[i+4,j+4] == 2:
                return 2

    for i in range(np.shape(board_matrix)[0]-4):
        for j in range(np.shape(board_matrix)[1]-4):
            if board_matrix[i+4,j] == 1 and board_matrix[i+3,j+1] == 1 and board_matrix[i+2,j+2] == 1 and board_matrix[i+1,j+3] == 1 and board_matrix[i,j+4] == 1:
                return 1
            elif board_matrix[i+4,j] == 2 and board_matrix[i+3,j+1] == 2 and board_matrix[i+2,j+2] == 2 and board_matrix[i+1,j+3] == 2 and board_matrix[i,j+4] == 2:
                return 2

    return 0

def pvp():
    turn = 0
    board_matrix = np.zeros((7,7),dtype='i')
    
    while True:
        print_board(board_matrix)
        winner = check_end(board_matrix)
        if winner == 1:
            print('총 {}수만에 흰 돌이 승리했습니다!'.format(turn))
            break
        if winner == 2:
            print('총 {}수만에 검은 돌이 승리했습니다!'.format(turn))
            break
        if turn%2 == 0:
            while True:
                x = int(input('검은 돌 차례: Row 입력(1~7)>'))
                y = int(input('Col 입력(1~7)>'))
                if 1<=x and x<=7 and 1<=y and y<=7:
                    if isempty(board_matrix,x,y):
                        break
            board_matrix = add_stone(board_matrix, x, y, 2)
            turn += 1
        else:
            while True:
                x = int(input('흰 돌 차례: Row 입력(1~7)>'))
                y = int(input('Col 입력(1~7)>'))
                if 1<=x and x<=7 and 1<=y and y<=7:
                    if isempty(board_matrix,x,y):
                        break
            board_matrix = add_stone(board_matrix, x, y, 1)
            turn += 1    

def pvAI(policy_wh, policy_bk, player_turn):
    turn = 0
    board_matrix = np.zeros((7,7),dtype='i')
    
    while True:
        print_board(board_matrix)

        winner = check_end(board_matrix)
        if winner != 0:
            if winner == player_turn:
                print('총 {}수만에 승리!'.format(turn))
                break
            else:
                print('총 {}수만에 패배!'.format(turn))
                break
        if player_turn == 1:
            policy = policy_bk
        elif player_turn == 2:
            policy = policy_wh

        if turn%2 == player_turn%2:
            while True:
                x = int(input('당신의 차례: Row 입력(1~7)>'))
                y = int(input('Col 입력(1~7)>'))
                if 1<=x and x<=7 and 1<=y and y<=7:
                    if isempty(board_matrix,x,y):
                        break
            board_matrix = add_stone(board_matrix, x, y, player_turn)
            turn += 1
        else:
            #board_matrix2 = translate_board(board_matrix)
            obs = mat_to_tensor(board_matrix) 
            out = policy(obs)
            m = Categorical(out)
            action, reward, dump_flag = make_action(board_matrix, m)
            if dump_flag:
                print('dump!')
                while True:
                    x = randint(1,7)
                    y = randint(1,7)
                    if isempty(board_matrix,x,y):
                        break
            else:
                x = int(action)//np.shape(board_matrix)[1] + 1
                y = int(action)%np.shape(board_matrix)[1] + 1
            board_matrix = add_stone(board_matrix, x, y, player_turn%2 + 1)
            turn += 1    

def AIvsAI(policy_wh, policy_bk, case):
    turn = 0
    board_matrix = np.zeros((7,7),dtype='i')
    
    while True:
        #print_board(board_matrix)

        winner = check_end(board_matrix)
        if winner != 0:
            if winner == 1 and case == 1:
                #print('총 {}수만에 흰돌(lev1) 승리!'.format(turn))
                return winner, turn
            elif winner == 1 and case == 2:
                #print('총 {}수만에 흰돌(lev2) 승리!'.format(turn))
                return winner, turn
            elif winner == 2 and case == 1:
                #print('총 {}수만에 검은돌(lev2) 승리!'.format(turn))
                return winner, turn
            elif winner == 2 and case == 2:
                #print('총 {}수만에 검은돌(lev1) 승리!'.format(turn))
                return winner, turn
        if turn == 49:
            #print('무승부!')
            return winner, turn

        #board_matrix2 = translate_board(board_matrix)
        obs = mat_to_tensor(board_matrix) 
        if turn%2 == 0:
            out = policy_bk(obs)
            m = Categorical(out)
            action, reward, dump_flag = make_action(board_matrix, m)
            if dump_flag:
                print('dump!')
                x = randint(1,7)
                y = randint(1,7)
                while not isempty(board_matrix,x,y):
                    x = randint(1,7)
                    y = randint(1,7)
            else:
                x = int(action)//np.shape(board_matrix)[1] + 1
                y = int(action)%np.shape(board_matrix)[1] + 1
            board_matrix = add_stone(board_matrix, x, y, 2)
            print_board(board_matrix)
            turn += 1    
        else:
            out = policy_wh(obs)
            m = Categorical(out)
            action, reward, dump_flag = make_action(board_matrix, m)
            if dump_flag:
                print('dump!')
                x = randint(1,7)
                y = randint(1,7)
                while not isempty(board_matrix,x,y):
                    x = randint(1,7)
                    y = randint(1,7)
            else:
                x = int(action)//np.shape(board_matrix)[1] + 1
                y = int(action)%np.shape(board_matrix)[1] + 1
            board_matrix = add_stone(board_matrix, x, y, 1)
            print_board(board_matrix)
            turn += 1    

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        self.gamma = 0.99
        
        self.fc1 = nn.Linear(49, 66)
        self.fc2 = nn.Linear(66, 49)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0005)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x
      
    def put_data(self, item):
        self.data.append(item)
        
    def replace_last_reward(self, reward):
        temp = self.data.pop()[1]
        self.data.append((reward,temp))

    def train(self):
        R = 0
        for r, log_prob in self.data[::-1]:
            R = r + R * self.gamma
            loss = -log_prob * R
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.data = []    

def make_action(board_matrix, m):
    iter = 0
    while True:
        action = m.sample()
        row = int(action)//np.shape(board_matrix)[1]
        col = int(action)%np.shape(board_matrix)[1]
        if isempty(board_matrix, row+1, col+1):
            dump_flag = False
            reward = 1.0
            break
        else:
            iter += 1
            if iter >= 20:  
                dump_flag = True
                reward = -10000.0
                break
    return action, reward, dump_flag

def mat_to_tensor(board_matrix):
    for i in range(np.shape(board_matrix)[0]):
        if i == 0:
            obs = board_matrix[i,:]
        else:
            obs = np.hstack((obs,board_matrix[i,:]))
    obs = torch.tensor(obs, dtype=torch.float)
    return obs
"""
def translate_board(board_matrix):
    board_matrix2 = board_matrix
    for i in range(np.shape(board_matrix)[0]):
        for j in range(np.shape(board_matrix)[1]):
            if board_matrix[i,j] == 2:
                board_matrix2[i,j] = -1
    return board_matrix2
"""
def makeAI(policy_wh, policy_bk, sheet, episode):
    for n_epi in range(episode):
        reward_wh = []
        reward_bk = []

        board_matrix = np.zeros((7,7),dtype='i')

        #print_board(board_matrix)

        turn = 0
        policy_wh.gamma = 0.99
        policy_bk.gamma = 0.99
        while True:
            #board_matrix2 = translate_board(board_matrix)
            obs = mat_to_tensor(board_matrix)
            done = False
            if turn%2==0:
                turn += 1
                out = policy_bk(obs)
                m = Categorical(out)
                action, reward, dump_flag = make_action(board_matrix, m)
                if dump_flag:
                    policy_bk.gamma = 0.00
                    policy_bk.put_data((reward, torch.log(out[action])))
                    reward_bk.append(reward)
                    break
                x = int(action)//np.shape(board_matrix)[1] + 1
                y = int(action)%np.shape(board_matrix)[1] + 1
                board_matrix = add_stone(board_matrix, x, y, 2)

                #print_board(board_matrix)

                if check_end(board_matrix) == 2:
                    reward = 40.0
                    policy_wh.replace_last_reward(-10.0)
                    done = True
                policy_bk.put_data((reward, torch.log(out[action])))
                reward_bk.append(reward)
            else:
                turn += 1
                out = policy_wh(obs)
                m = Categorical(out)
                action, reward, dump_flag = make_action(board_matrix, m)
                if dump_flag:
                    policy_wh.gamma = 0.00
                    policy_wh.put_data((reward, torch.log(out[action])))
                    reward_wh.append(reward)
                    break
                x = int(action)//np.shape(board_matrix)[1] + 1
                y = int(action)%np.shape(board_matrix)[1] + 1
                board_matrix = add_stone(board_matrix, x, y, 1)

                #print_board(board_matrix)

                if check_end(board_matrix) == 1:
                    reward = 40.0
                    policy_bk.replace_last_reward(-10.0)
                    done = True
                policy_wh.put_data((reward, torch.log(out[action])))
                reward_wh.append(reward)
            if done or turn == 49:
                break
        
        sum_reward_wh = 0
        sum_reward_bk = 0
        for i in range(len(reward_wh)):
            sum_reward_wh += reward_wh[i]
        for i in range(len(reward_bk)):
            sum_reward_bk += reward_bk[i]
        sheet.append([sum_reward_wh, sum_reward_bk, turn])

        #print_board(board_matrix)

        policy_wh.train()
        policy_bk.train()
        print('# of episode = {}, Turn = {}, dump_flag = {}'.format(n_epi+1, turn, dump_flag))

    return policy_wh, policy_bk


def main():
    wb = openpyxl.Workbook()
    sheet_lev1 = wb.active
    sheet_lev1.title = 'AI Lev1'
    sheet_lev2 = wb.create_sheet('AI Lev2')

    existAI_l1 = False
    existAI_l2 = False
    eter_l1 = 0
    eter_l2 = 0
    episode = 200
    while True:
        menu = int(input('1:player vs player /// 2: vs AI(level1) /// 3: vs AI(level2) /// 4: AI(lev1) vs AI(lev2) /// 5:finish '))
        if menu == 1:
            pvp()
            continue
        elif menu == 2:
            if not existAI_l1:
                policy_wh = Policy()
                policy_bk = Policy()
                policy_wh, policy_bk = makeAI(policy_wh, policy_bk, sheet_lev1, episode*50)
                eter_l1 += episode*50
                existAI_l1 = True
            if existAI_l1:
                while True:
                    train_more = input('더 학습시키겠습니까? 현재 학습 횟수 = {}게임 (y,n) '.format(eter_l1))
                    if train_more == 'y':
                        policy_wh, policy_bk = makeAI(policy_wh, policy_bk, sheet_lev1, episode*50)
                        eter_l1 += episode*50
                    else:
                        break
            player_turn = 0
            while player_turn != 1 and player_turn != 2:
                player_turn = int(input('you want 흰돌: 1, 검은돌: 2 '))
            pvAI(policy_wh, policy_bk, player_turn)
            continue
        elif menu == 3:
            if not existAI_l2:
                policy_wh1 = Policy()
                policy_bk1 = Policy()
                policy_wh2 = Policy()
                policy_bk2 = Policy()
                for i in range(25):
                    policy_wh1, policy_bk1 = makeAI(policy_wh1, policy_bk1, sheet_lev2, episode)
                    policy_wh2, policy_bk2 = makeAI(policy_wh2, policy_bk2, sheet_lev2, episode)
                    policy_wh1, policy_bk2 = makeAI(policy_wh1, policy_bk2, sheet_lev2, episode)
                    policy_wh2, policy_bk1 = makeAI(policy_wh2, policy_bk1, sheet_lev2, episode)
                eter_l2 += episode*50
                existAI_l2 = True
            if existAI_l2:
                while True:
                    train_more = input('더 학습시키겠습니까? 현재 학습 횟수 = {}게임 (y,n) '.format(eter_l2))
                    if train_more == 'y':
                        for i in range(25):
                            policy_wh1, policy_bk1 = makeAI(policy_wh1, policy_bk1, sheet_lev2, episode)
                            policy_wh2, policy_bk2 = makeAI(policy_wh2, policy_bk2, sheet_lev2, episode)
                            policy_wh1, policy_bk2 = makeAI(policy_wh1, policy_bk2, sheet_lev2, episode)
                            policy_wh2, policy_bk1 = makeAI(policy_wh2, policy_bk1, sheet_lev2, episode)
                        eter_l2 += episode*50
                    else:
                        break
            player_turn = 0
            while player_turn != 1 and player_turn != 2:
                player_turn = int(input('you want 흰돌: 1, 검은돌: 2 '))
            pvAI(policy_wh1, policy_bk1, player_turn)
            continue
        elif menu ==4:
            if (not existAI_l1) or (not existAI_l2):
                print('먼저 학습시키십시오.')
                continue
            case = 0
            while case != 1 and case != 2:
                case = int(input('1: 흰돌(lev1) vs 검은돌(lev2) /// 2: 흰돌(lev2) vs 검은돌(lev1)'))
                wh_win = 0
                bk_win = 0
                draw = 0
                avg_turn = 0
                if case == 1:
                    for i in range(10):
                        winner, turn = AIvsAI(policy_wh, policy_bk1, case)
                        if winner == 1:
                            wh_win += 1
                        elif winner == 2:
                            bk_win += 1
                        else:
                            draw += 1
                        avg_turn += turn
                    print('흰돌 {}승, 검은돌 {}승, 무승부 {}회, 평균 {}수만에 승부가 결정났습니다.'.format(wh_win, bk_win, draw, avg_turn/10))
                elif case == 2:
                    for i in range(10):
                        winner, turn = AIvsAI(policy_wh1, policy_bk, case)
                        if winner == 1:
                            wh_win += 1
                        elif winner == 2:
                            bk_win += 1
                        else:
                            draw += 1
                        avg_turn += turn
                    print('흰돌 {}승, 검은돌 {}승, 무승부 {}회, 평균 {}수만에 승부가 결정났습니다.'.format(wh_win, bk_win, draw, avg_turn/10))

        elif menu == 5:
            wb.save('result.xlsx')
            #uploaded = files.upload()
            #files.download('result.xlsx')
            break
        
main()