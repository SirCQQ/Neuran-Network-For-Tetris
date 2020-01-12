import numpy as np
import random as rand
import copy
import time
import datetime
import math
import random
from os import system
import cv2
from PIL import Image
from keras import Sequential, layers, optimizers
from collections import deque


def clear():
    system("clear")


class Tetris:
    HEIGHT = 20
    WIDTH = 10
    PIECE = {
        0: {  # I
            0: [(0, 0), (1, 0), (2, 0), (3, 0)],
            1: [(0, 0), (0, 1), (0, 2), (0, 3)],
            2: [(0, 0), (1, 0), (2, 0), (3, 0)],
            3: [(0, 0), (0, 1), (0, 2), (0, 3)],
        },


        # 0:{0: [(0, 0), (0, 0), (0, 0), (0, 0)],
        # 1: [(0, 0), (0, 0), (0, 0), (0, 0)],
        # 2: [(0, 0), (0, 0), (0, 0), (0, 0)],
        # 3: [(0, 0), (0, 0), (0, 0), (0, 0)]},



        1: {  # T
            0: [(1, 0), (0, 1), (1, 1), (2, 1)],
            1: [(0, 1), (1, 2), (1, 1), (1, 0)],
            2: [(1, 2), (2, 1), (1, 1), (0, 1)],
            3: [(2, 1), (1, 0), (1, 1), (1, 2)],
        },
        2: {  # L
            0: [(1, 0), (1, 1), (1, 2), (2, 2)],
            1: [(0, 1), (1, 1), (2, 1), (2, 0)],
            2: [(1, 2), (1, 1), (1, 0), (0, 0)],
            3: [(2, 1), (1, 1), (0, 1), (0, 2)],
        },
        3: {  # J
            0: [(1, 0), (1, 1), (1, 2), (0, 2)],
            1: [(0, 1), (1, 1), (2, 1), (2, 2)],
            2: [(1, 2), (1, 1), (1, 0), (2, 0)],
            3: [(2, 1), (1, 1), (0, 1), (0, 0)],
        },
        4: {  # Z
            0: [(0, 0), (1, 0), (1, 1), (2, 1)],
            1: [(0, 2), (0, 1), (1, 1), (1, 0)],
            2: [(0, 0), (1, 0), (1, 1), (2, 1)],
            3: [(0, 2), (0, 1), (1, 1), (1, 0)],
        },
        5: {  # S
            0: [(2, 0), (1, 0), (1, 1), (0, 1)],
            1: [(0, 0), (0, 1), (1, 1), (1, 2)],
            2: [(2, 0), (1, 0), (1, 1), (0, 1)],
            3: [(0, 0), (0, 1), (1, 1), (1, 2)],
        },
        6: {  # O
            0: [(0, 0), (1, 0), (0, 1), (1, 1)],
            1: [(0, 0), (1, 0), (0, 1), (1, 1)],
            2: [(0, 0), (1, 0), (0, 1), (1, 1)],
            3: [(0, 0), (1, 0), (0, 1), (1, 1)],
        }
    }
    COLORS = {
        0: (255, 255, 255),
        1: (247, 64, 99),
        2: (0, 167, 247),
        # 2: (0,0, 0),

        # 2:(rand.randint(0,250),rand.randint(0,250),rand.randint(0,250))
    }

    def __init__(self):
        self.train_frames = 0
        self.observe = 1000
        self.no_frames_to_save = 0
        self.no_frames_between_trains = 0

        self.epsilon = 0.99  # exploration rate
        self.gamma = 0.9  # discount rate
        self.reduce_epsilon = 0.000001  # 10^(-6)

        self.input_layer_size = 4
        self.batch_size = 0
        self.epochs = 0

        self.model = None

        self.parameters = {}
        self.reset()

    def reset(self):
        '''Reset game to empty table and score 0'''
        self.board = [[0 for _ in range(self.WIDTH)]
                      for __ in range(self.HEIGHT)]
        self.score = 0
        self.game_over = False
        self.next_piece = self.get_next_piece()
        self.piece_rotation = 0
        self.x_start = 0
        self.y_start = int(self.WIDTH/2)-2
        self.reach_bottom = False

    def get_next_piece(self):
        aux = copy.deepcopy(self.PIECE)
        rand.shuffle(aux)
        self.piece_rotation = 0
        return aux.pop(0)
        # for fun get just little cubs with other commented propriety
        # return self.PIECE[0]

    # Verify colosion down
    def colision_down(self, rotation, directionY):
        piece = self.next_piece[(self.piece_rotation+rotation) % 4]
        for i in range(4):
            x = self.x_start+piece[i][0]+1
            y = self.y_start+piece[i][1]+directionY
            if(x > 19 or self.board[x][y] == 2 or y < 0 or y > 9):
                return True

        return False

    # Verify colision left or right
    def colisions_sides(self, rotation, directionY):
        piece = self.next_piece[(self.piece_rotation+rotation) % 4]
        for i in range(4):
            x = self.x_start+piece[i][0]
            y = self.y_start+piece[i][1]+directionY
            if(y > 9 or y < 0 or x > 19 or self.board[x][y] == 2):
                return True
        return False

    # Check colision on rotation
    def colision_roatate(self, rotation):
        piece = self.next_piece[(self.piece_rotation+rotation) % 4]
        for i in range(4):
            x = self.x_start+piece[i][0]
            y = self.y_start+piece[i][1]
            if(y > 9 or y < 0 or x > 19 or self.board[x][y] == 2):
                print(x, y)
                return True

        return False

    # clear previous state and redraw with update rotation and XoY
    def draw_piece_on_board(self, rotation, directionX, directionY):
        piece = self.next_piece[self.piece_rotation]
        for i in range(self.HEIGHT):
            for j in range(self.WIDTH):
                if(self.board[i][j] == 1):
                    self.board[i][j] = 0
        piece = self.next_piece[(self.piece_rotation+rotation) % 4]
        for i in range(4):
            x = self.x_start+piece[i][0]+directionX
            y = self.y_start+piece[i][1]+directionY
            self.board[x][y] = 1

    # When hit ground change from 1 (moving piece) to 2 (solid one )

    def make_solid(self):
        for i in range(self.HEIGHT):
            if 1 in self.board[i]:
                for j in range(self.WIDTH):
                    if(self.board[i][j] == 1):
                        self.board[i][j] = 2
    # rotaion is -1,0,1 direction same

    # chekc if top layer has any solid blocks if yes game over
    def check_game_over(self):
        if(2 in self.board[0]):
            time.sleep(0.2)
            return True
        else:
            return False

    # check for complet rows,clear and return the number of rows cleard for score
    def check_complete_row(self):
        line = 0
        for i in range(self.HEIGHT):
            if 0 not in self.board[i]:
                self.board.pop(i)
                self.board.insert(0, [0 for k in range(self.WIDTH)])
                line += 10
        return line

    # When hit's something on X stop make it solid and change position of start with new shape
    def next_state(self):
        self.x_start = 0
        self.y_start = int(self.WIDTH/2)-2
        self.make_solid()
        self.next_piece = self.get_next_piece()
        bonus = self.check_complete_row()
        self.score += 10 + ((bonus*10))

    @staticmethod
    def euclidian_distance(ax, ay, bx, by):
        return np.around(np.array([math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)]), decimals=1)

    @staticmethod
    def manhattan_distance(ax, ay, bx, by):
        return np.around(np.array([abs(ax - bx) + abs(ay - by)]), decimals=1)

    def save_model(self):
        today = datetime.datetime.today()
        file_name = 'day_' + str(today.day - 17) + '_saved_' + str(today.hour) + '_' + str(
            today.minute) + '_dnq.h5'
        self.model.save_weights(file_name)
        print("Model", file_name, "salvat!", sep=" ")

    def build_model(self, file_saved_weights=''):
        model = Sequential()

        model.add(layers.Dense(self.parameters['dimension_layer1'],
                               activation=self.parameters['activation_layer1'],
                               kernel_initializer='glorot_normal',
                               input_shape=(self.input_layer_size,)))  # lecun_uniform #ceva random
        model.add(layers.Dense(self.parameters['dimension_layer2'],
                               activation=self.parameters['activation_layer2'],
                               kernel_initializer='glorot_normal'))

        model.add(layers.Dense(
            4, activation=self.parameters['activation_layer3']))

        model.compile(optimizer=optimizers.Adam(lr=1e-2),
                      loss='mean_squared_error', metrics=['accuracy'])

        if file_saved_weights != '':
            model.load_weights(file_saved_weights)
        return model

    def train_net(self, train_frames=0, batch_size=0, parameters=None, no_frames_to_save=0, epochs=0, no_frames_between_trains=0):
        if train_frames != 0:
            self.train_frames = train_frames
        if batch_size != 0:
            self.batch_size = batch_size
        if parameters is not None:
            self.parameters = parameters
        if no_frames_to_save != 0:
            self.no_frames_to_save = no_frames_to_save
        if epochs != 0:
            self.epochs = epochs
        if no_frames_between_trains != 0:
            self.no_frames_between_trains = no_frames_between_trains

        self.model = self.build_model()

        replay = deque(maxlen=self.observe)
        current_state = env.board

        no_frame = 0
        for no_frame in range(self.train_frames):
            print(no_frame, self.train_frames)
            if self.check_game_over():
                self.reset()

            if random.random() < self.epsilon or no_frame < self.observe:
                if random.random() < 0.5:
                    action = env.play(rand.randint(-1, 1), 1)
                else:
                    action = env.play(rand.randint(-1, 1), rand.randint(-1, 1))
            else:
                action = env.play(rand.randint(-1, 1), 0)
            env.render()

            new_state = env.board

            if len(replay) == self.observe:
                replay.popleft()

            replay.append((current_state, action, env.score, new_state))

            if (no_frame > self.observe) and (no_frame % self.no_frames_between_trains == 0):
                minibatch = random.sample(replay, self.batch_size)
                X_train, Y_train = self.process_minibatch(minibatch)

                # batch_size #epochs
                self.model.fit(X_train, Y_train, epochs=self.epochs)

            current_state = new_state

            if self.epsilon > 0.1 and no_frame > self.observe:
                self.epsilon -= self.reduce_epsilon
            print(no_frame, self.no_frames_to_save)
            if (no_frame != 0) and (no_frame % self.no_frames_to_save == 0):
                self.save_model()
            no_frame += 1
        self.save_model()

    
    def process_minibatch(self, minibatch):
        len_minibatch = len(minibatch)

        old_states_replay = np.zeros((len_minibatch, self.input_layer_size))
        actions_replay = np.zeros((len_minibatch,))
        rewards_replay = np.zeros(len_minibatch, )
        new_states_replay = np.zeros((len_minibatch, self.input_layer_size))

        for index, memory in enumerate(minibatch):
            old_state_mem, action_mem, reward_mem, new_state_mem = memory
            if action_mem == 119:  # up
                actions_replay[index] = 0
            elif action_mem == 97:  # left
                actions_replay[index] = 1
            elif action_mem == 100:  # right
                actions_replay[index] = 2
            elif action_mem == 115:  # down
                actions_replay[index] = 3
            rewards_replay[index] = reward_mem

        old_qvals = self.model.predict(old_states_replay)
        new_qvals = self.model.predict(new_states_replay)

        maxQs = np.max(new_qvals, axis=1)

        target = old_qvals


        return old_states_replay, target

    def play(self, rotation, directionY):
        if self.check_game_over() == True:
            self.reset()
        directionX = 1
        # Test if block can rotate
        if (self.colision_roatate(rotation)) == True:
            rotation = 0
        # Test if block with rotation can move right or left
        if self.colisions_sides(rotation, directionY) == True:
            directionY = 0
        # Test if block with rotation and right/left move can go down
        if self.colision_down(rotation, directionY) == True:
            directionX = 0
        # Draw with checked rotation movement
        self.draw_piece_on_board(rotation, directionX, directionY)
        # update values
        self.x_start += directionX
        self.y_start += directionY
        self.piece_rotation = (self.piece_rotation+rotation) % 4
        # Get next piece on board
        if(directionX == 0):
            self.next_state()

        return self.board, self.score

    def render(self):
        # CAREFULL
        # For epileptic and insame mod delete Tetris from "Tetris.Colors"
        COLORS = {
            # 0: (255, 255, 255),
            # 1: (247, 64, 99),
            0: (rand.randint(0, 250), rand.randint(0, 250), rand.randint(0, 250)),

            1: (rand.randint(0, 250), rand.randint(0, 250), rand.randint(0, 250)),
            # 2: (0, 167, 247),
            2: (rand.randint(0, 250), rand.randint(0, 250), rand.randint(0, 250))
        }
        '''Renders the current board'''
        img = [Tetris.COLORS[p] for row in self.board for p in row]
        img = np.array(img).reshape(
            Tetris.HEIGHT, Tetris.WIDTH, 3).astype(np.uint8)
        img = img[..., ::-1]  # Convert RRG to BGR (used by cv2)
        img = Image.fromarray(img, 'RGB')
        img = img.resize((Tetris.WIDTH * 50, Tetris.HEIGHT * 50))
        img = np.array(img)
        # cv2.putText(img, str(self.score), (22, 22), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        cv2.imshow('image', np.array(img))
        cv2.waitKey(1)
        # time.sleep(10)


env = Tetris()
# print(env.colision())
maxScore = -1
params = {
    "dimension_layer1": 128,
    "activation_layer1": "relu",
    "dimension_layer2": 32,
    "activation_layer2": "relu",
    "activation_layer3": "linear"
}
for j in range(10000):
    # info=env.play(rand.randint(-1,1),1)
    # info = env.play(rand.randint(-1, 1), rand.randint(-1, 1))
    env.train_net(batch_size=64, train_frames=150000, parameters=params,
                  no_frames_to_save=1000, epochs=1, no_frames_between_trains=100)

    # info=env.play(0,rand.randint(-1,1))
    print(env.score)
    env.render()
    if (j+1) % 100 == 0:
        print(j+1)
print(maxScore)
