import json
import random

import pygame, sys, math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten
from keras.models import model_from_json
from keras.optimizers import RMSprop
from skimage.color import rgb2grey
from collections import deque

black = (0, 0, 0)
red = (255, 0, 0)
blue = (0, 0, 255)
white = (255, 255, 255)

colors = [((6-j)*40, j*40, j*40) for j in range(6)]

LOAD_FILE = False

tw = 84
th = 84


class QLearning:
    def __init__(self, num_actions=3, load=False):
        self.load = load
        self.epsilon = 1.0
        self.start_train_frames = 1000
        self.max_memory = 40000
        self.batch_size = 32
        self.model = None
        self.num_actions = num_actions
        self.memory = deque()
        self.discount = 0.99
        self.loss = 0
        self.conv_model()
        self.t = 0
        self.epsilon_step = 0.9 / 10000

    def conv_model(self):
        if self.load is True:
            with open("model.json", "r") as jfile:
                self.model = model_from_json(json.load(jfile))
            self.model.load_weights("model.h5")
            rmsprop = RMSprop(lr=0.00025, rho=0.9, epsilon=1e-08, decay=0.95)
            self.model.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['accuracy'])
            print("model loaded!")
        else:
            self.model = Sequential()
            self.model.add(Convolution2D(32, 4, 4, subsample=(4, 4), init="normal", input_shape=(4, tw, th), dim_ordering='th'))
            self.model.add(Activation("relu"))
            self.model.add(Convolution2D(64, 4, 4, subsample=(2, 2), init="normal", activation='relu', dim_ordering='th'))
            self.model.add(Convolution2D(64, 3, 3, subsample=(1, 1), init="normal", activation='relu', dim_ordering='th'))
            self.model.add(Flatten())
            self.model.add(Dense(512, init="normal", activation='relu'))
            self.model.add(Dense(self.num_actions, init="normal"))
            rmsprop = RMSprop(lr=0.00025, rho=0.9, epsilon=1e-08, decay=0.95)
            self.model.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['accuracy'])

    def random_action(self):
        return np.random.randint(0, 3, size=1)

    def get_test_action(self, state):
        if np.random.rand() <= 0.1:
            action = np.random.randint(0, 3, size=1)
        else:
            q = self.model.predict(np.array([state]))
            action = np.argmax(q[0])
        return action

    def get_action(self, state):
        if LOAD_FILE is True:
            if np.random.rand() <= 0.1:
                action = np.random.randint(0, 3, size=1)
            else:
                q = self.model.predict(np.array([state]))
                action = np.argmax(q[0])
            return action

        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, 3, size=1)
        else:
            q = self.model.predict(np.array([state]))
            action = np.argmax(q[0])
            if self.t % 100 == 0:
                print(q[0])

        if self.epsilon > 0.1:
            self.epsilon -= self.epsilon_step

        return action

    def replay(self, data):
        if len(self.memory) + 1 > self.max_memory:
            self.memory.popleft()
        self.memory.append(data)

    def get_batch(self):
        batches = random.sample(self.memory, self.batch_size)
        inputs = []
        actions = []
        rewards = []
        new_states = []
        game_over = []
        for b in batches:
            inputs.append(b[0])
            actions.append(b[1])
            rewards.append(b[2])
            new_states.append(b[3])
            game_over.append(b[4])

        inputs = np.array(inputs)
        new_states = np.array(new_states)
        targets = self.model.predict(inputs)
        qs = self.model.predict(new_states)
        for i in range(len(batches)):
            if game_over[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.discount * np.max(qs[i])
        return inputs, targets

    def train(self, data):
        self.replay(data)
        loss = 0
        if self.t % 4 == 0:
            for i in range(4):
                inputs, targets = self.get_batch()
                loss = self.model.train_on_batch(inputs, targets)[0]
        if self.t % 1000 == 0:
            self.model.save_weights("model.h5", overwrite=True)
            with open("model.json", "w") as outfile:
                json.dump(self.model.to_json(), outfile)

        self.t += 1
        return loss


def weightedAverage(pixel):
    return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]


class BreakoutEnv:
    def change(self):
        last_score = self.score["score"]
        last_lives = self.score["lives"]
        seconds = (pygame.time.get_ticks() - self.start_ticks) / 1000
        if seconds >= 1:
            self.start_ticks = pygame.time.get_ticks()
        return last_score, last_lives, seconds

    def observe(self):
        reward = self.score["score"]
        reward -= 2*self.score["lives"]
        game_over = False
        if self.score["lives"] == 0:
            # reward = -100
            game_over = True
        return reward, game_over

    def init_state_frame(self):
        arr = pygame.surfarray.pixels3d(self.pixel_state)
        grey = np.zeros((tw, th), dtype=np.float32)
        for rownum in range(len(arr)):
            for colnum in range(len(arr[rownum])):
                grey[rownum][colnum] = weightedAverage(arr[rownum][colnum])
        state = [grey for _ in range(4)]
        return np.array(state) / 255.0

    def get_frame(self):
        arr = pygame.surfarray.pixels3d(self.pixel_state)
        grey = np.zeros((tw, th), dtype=np.float32)
        for rownum in range(len(arr)):
            for colnum in range(len(arr[rownum])):
                grey[rownum][colnum] = weightedAverage(arr[rownum][colnum])
        state = [grey]
        return np.array(state) / 255.0

    def act(self, action):
        for i in range(4):
            self.step(action)

    def train_agent(self):
        agent = QLearning(load=LOAD_FILE)
        
        self.initialise()
        state = self.init_state_frame()
        for random_steps in range(agent.start_train_frames):
            action = agent.random_action()
            self.act(action)
            reward, game_over = self.observe()
            next_state = self.get_frame()
            next_state = np.append(state[1:, :, :], next_state, axis=0)
            agent.replay((state, action, reward, next_state, game_over))
            if game_over:
                self.initialise()

        for e in range(50):
            game_over = False
            rewardless = 0
            loss = 0
            for train_step in range(100000):
                action = agent.get_action(state)
                self.act(action)
                reward, game_over = self.observe()
                next_state = self.get_frame()
                next_state = np.append(state[1:, :, :], next_state, axis=0)
                loss += agent.train((state, action, reward, next_state, game_over))
                state = next_state
                if game_over:
                    print("epsilon {:.4f}, loss {:.4f} , reward {}".format(agent.epsilon, loss, reward))
                    if reward < 2:
                        rewardless += 1
                    if rewardless > 5 and agent.epsilon < 0.12:
                        rewardless = 0
                        agent.epsilon = 1.0
                    self.initialise()
            print("episode : ", e)

    def play(self):
        self.initialise()
        agent = QLearning(load=True)
        game_over = False
        state = self.init_state_frame()
        for i in range(1000):
            action = agent.get_test_action(state)
            self.act(action)
            reward, game_over = self.observe()
            next_state = self.get_frame()
            next_state = np.append(state[1:, :, :], next_state, axis=0)
            if game_over:
                self.initialise()
            state = next_state

    def step(self, action=None):
        # msElapsed = self.clock.tick(144)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if action == 0:
            self.paddle[0] -= self.speed
            self.paddle[0] = max(0, self.paddle[0])
        if action == 2:
            self.paddle[0] += self.speed
            self.paddle[0] = min(self.w - self.paddle[2], self.paddle[0])

        if self.score["lives"] == 0:
            pass
        else:
            self.update()

        self.draw()

    def draw(self):
        self.screen.fill(black)
        pygame.draw.rect(self.screen, (255, 100, 0), self.paddle, 0)
        for shape in self.entities:
            pygame.draw.rect(self.screen, colors[shape[-1]], shape[:-1], 0)
            # pygame.draw.rect(self.screen, black, shape, 1)
        new_ball = [int(self.ball[0]), int(self.ball[1])]
        pygame.draw.circle(self.screen, white, new_ball, self.radius)
        pygame.display.update()
        pygame.transform.scale(self.screen, (tw, th), self.pixel_state)
        # write the score and lives
        '''myfont = pygame.font.SysFont("monospace", 15)
        label = myfont.render("Score:" + str(self.score["score"]), 1, (255, 255, 0))
        self.screen.blit(label, (0, 0))
        label = myfont.render("Lives:" + str(self.score["lives"]), 1, (255, 255, 0))
        self.screen.blit(label, (100, 0))'''

    def update(self):
        self.prev = np.array(self.ball)
        self.ball_vel = np.clip(self.ball_vel, -4, 4)
        self.ball += self.ball_vel
        self.ball_collision()

    def ball_collision(self):
        ball = self.ball
        ball_vel = self.ball_vel
        radius = self.radius
        if ball[0] - radius < 0 or ball[0] + radius > self.w:
            ball_vel[0] *= -1
            return
        if ball[1] - radius < 0:
            ball_vel[1] *= -1
            return
        if ball[1] > self.h:
            temp = [-3, 3]
            self.ball_vel = np.array([temp[random.randint(0,1)], 1])
            self.ball = list(self.ball_initial)
            self.paddle = list(self.paddle_initial)
            self.score["lives"] -= 1
            return

        # paddle collision
        self.paddle_collision = False
        if ball[0] + radius < self.paddle[0] \
                or ball[0] - radius > self.paddle[0] + self.paddle[2] \
                or ball[1] + radius < self.paddle[1] \
                or ball[1] - radius > self.paddle[1] + self.paddle[3]:
            pass
        else:
            self.paddle_collision = True
            paddle = self.paddle
            if self.prev[1] + radius < paddle[1] \
                    or self.prev[1] - radius > paddle[1] + paddle[3]:
                if ball[0] < paddle[0] + 0.1 * paddle[2] or ball[0] > paddle[0] + 0.9 * self.paddle[2]:
                    ball_vel[0] *= -1
                    ball_vel[1] = 1
                    if ball_vel[0] < 0:
                        ball_vel[0] = -2
                    else:
                        ball_vel[0] = 2
                ball_vel[1] *= -1
            else:
                ball_vel[0] *= -1

        # bricks collision
        for brick in self.entities:
            if ball[0] + radius < brick[0] or ball[0] - radius > brick[0] + brick[2] \
                    or ball[1] + radius < brick[1] \
                    or ball[1] - radius > brick[1] + brick[3]:
                pass
            else:
                reverse_vel = [np.sign(-v) for v in ball_vel]
                for i in range(10):
                    older = (ball[0] + i * reverse_vel[0], ball[1] + i * reverse_vel[1])
                    if older[1] + radius < brick[1] or older[1] - radius > brick[1] + brick[3]:
                        ball_vel[1] *= -1
                        break
                    elif older[0] + radius < brick[0] or older[0] - radius > brick[0] + brick[2]:
                        ball_vel[0] *= -1
                        break
                # ball_vel[0] += ball_vel[0] * 0.1
                # ball_vel[1] += ball_vel[1] * 0.1
                brick[0] = self.w + 100
                self.score["score"] += (7-brick[4])

    def __init__(self):
        self.w, self.h = 480, 480
        pw = 0.20 * self.w
        ph = 0.02 * self.h
        bw = self.w / 10
        bh = 16
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.w, self.h), 0, 32)
        pygame.display.set_caption('Breakout')
        pygame.mouse.set_visible(0)
        # game objects
        self.speed = 5
        self.radius = 6
        self.score = dict()
        self.entities_initial = [[i * bw, (j + 5) * bh, bw, bh] for i in range(10) for j in range(5)]
        self.ball = np.array([0, 0])
        self.ball_initial = (int(self.w / 2 - self.radius/2), int(self.h / 2 - 20))
        self.ball_vel = np.array([0, 0])
        ball_surface = pygame.Surface((2 * self.radius, 2 * self.radius))
        ball_surface.fill(black)
        ball_surface.set_colorkey(black)
        pygame.draw.circle(ball_surface, (255, 255, 255), (self.radius, self.radius), self.radius)
        ball_surface.set_alpha(50)
        self.ball_surface = ball_surface
        self.paddle_initial = (self.w / 2 - pw / 2, self.h - ph, pw, ph)
        self.paddle = [0, 0, 0, 0]
        self.prev = np.array([0, 0])
        self.entities = None
        self.timer = 0
        self.start_ticks = 0
        self.pixel_state = pygame.Surface((tw, th))
        self.surf = None
        self.paddle_collision = False
        self.max_lives = 5

    def initialise(self):
        self.score["lives"] = self.max_lives
        self.score["score"] = 0
        self.start_ticks = pygame.time.get_ticks()
        self.paddle = list(self.paddle_initial)
        self.ball = np.array(self.ball_initial, dtype=np.float32)
        temp = [-3, 3]
        self.ball_vel = np.array([temp[random.randint(0,1)], 1])
        bw = self.w / 10
        bh = 16
        self.entities = [[i * bw, (j + 5) * bh, bw, bh, j] for i in range(10) for j in range(6)]


if __name__ == "__main__":
    br = BreakoutEnv()
    br.train_agent()
