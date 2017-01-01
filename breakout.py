import json
import random

import pygame, sys, math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten
from keras.models import model_from_json
from skimage.color import rgb2gray

black = (0, 0, 0)
red = (255, 0, 0)
blue = (0, 0, 255)
white = (255, 255, 255)


class QLearning:
    def __init__(self, num_actions=3):
        self.epsilon = 1.0
        self.train_timestep = 4
        self.random_replay_frames = 20000
        self.max_memory = 100000
        self.batch_size = 32
        self.model = None
        self.num_actions = num_actions
        self.memory = []
        self.discount = 0.99
        self.loss = 0
        self.conv_model()
        self.t = 0

    def conv_model(self):
        self.model = Sequential()
        self.model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(4, 84, 84), dim_ordering='th'))
        self.model.add(Activation("relu"))
        self.model.add(Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu'))
        self.model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(self.num_actions))
        self.model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

    def get_action(self, state):
        if np.random.rand() <= self.epsilon or self.t < self.random_replay_frames:
            action = np.random.randint(0, self.num_actions, size=1)
        else:
            q = self.model.predict(np.reshape(state, (1, 4, 84, 84)))
            action = np.argmax(q[0])

        if self.epsilon > 0.1 and self.t >= self.random_replay_frames:
            self.epsilon -= 9e-7

        return action

    def replay(self, data):
        if len(self.memory) + 1 > self.max_memory:
            del self.memory[0]
        self.memory.append(data)

    def get_batch(self):
        memory_size = len(self.memory)
        batch_indices = np.random.randint(0, memory_size, self.batch_size)
        inputs = np.float32(np.array([self.memory[i][0] for i in batch_indices]) / 255.0)

        targets = np.zeros((self.batch_size, self.num_actions))
        for i, idx in enumerate(batch_indices):
            state, action, reward, new_state, game_over = self.memory[idx]
            targets[i] = self.model.predict(np.float32(np.reshape(state, (1, 4, 84, 84)) / 255.0))[0]
            q = np.max(self.model.predict(np.float32(np.reshape(new_state, (1, 4, 84, 84)) / 255.0))[0])
            if game_over:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount * q
        # inputs = np.reshape(inputs, [self.batch_size, 4, 84, 84])
        return inputs, targets

    def train(self, data):
        self.replay(data)

        if self.t >= self.random_replay_frames:
            if self.t % self.train_timestep == 0:
                inputs, targets = self.get_batch()
                self.loss = self.model.train_on_batch(inputs, targets)
            if self.t % 300000 == 0:
                self.model.save_weights("model.h5", overwrite=True)
                with open("model.json", "w") as outfile:
                    json.dump(self.model.to_json(), outfile)

        self.t += 1


class BreakoutEnv:
    def change(self):
        last_score = self.score["score"]
        last_lives = self.score["lives"]
        seconds = (pygame.time.get_ticks() - self.start_ticks) / 1000
        if seconds >= 1:
            self.start_ticks = pygame.time.get_ticks()
        return last_score, last_lives, seconds

    def observe(self, score, lives, seconds):
        reward = 0
        if self.score["score"] - score > 0:
            reward = 1
        if self.score["lives"] - lives < 0:
            reward = -1
        game_over = False
        if self.score["score"] >= len(self.entities_initial):
            reward = 1
            game_over = True
        if self.score["lives"] <= 0:
            reward = -1
            game_over = True
        return reward, game_over

    def init_state_frame(self):
        pygame.transform.scale(self.screen, (84, 84), self.pixel_state)
        arr = pygame.surfarray.pixels3d(self.pixel_state)
        processed_observation = np.uint8(rgb2gray(arr) * 255)
        state = [processed_observation for _ in range(4)]
        return np.stack(state, axis=0)

    def get_frame(self):
        pygame.transform.scale(self.screen, (84, 84), self.pixel_state)
        arr = pygame.surfarray.pixels3d(self.pixel_state)
        arr = np.uint8(rgb2gray(arr) * 255)
        return np.reshape(arr, (1, 84, 84))

    def train_agent(self):
        agent = QLearning()
        for e in range(12000):
            game_over = False
            self.initialise()
            state = self.init_state_frame()
            while not game_over:
                score, lives, seconds = self.change()
                action = agent.get_action(state)
                [self.step(action) for _ in range(4)]
                reward, game_over = self.observe(score, lives, seconds)
                next_state = self.get_frame()
                next_state = np.append(state[1:, :, :], next_state, axis=0)
                agent.train((state, action, reward, next_state, game_over))
                state = next_state

    def play(self):
        self.initialise()
        with open("model.json", "r") as jfile:
            model = model_from_json(json.load(jfile))
        model.load_weights("model.h5")
        model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
        game_over = False
        state = self.init_state_frame()
        while not game_over:
            q = model.predict(np.reshape(state, (1, 4, 84, 84)))
            action = np.argmax(q[0])
            [self.step(action) for _ in range(4)]
            next_state = self.get_frame()
            next_state = np.append(state[1:, :, :], next_state, axis=0)
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
            self.initialise()
        else:
            self.update()

        self.draw()

    def draw(self):
        self.screen.fill(black)
        pygame.draw.rect(self.screen, red, self.paddle, 0)
        for shape in self.entities:
            pygame.draw.rect(self.screen, blue, shape, 0)
            pygame.draw.rect(self.screen, black, shape, 1)
        new_ball = [int(self.ball[0]), int(self.ball[1])]
        pygame.draw.circle(self.screen, white, new_ball, self.radius)
        pygame.display.update()
        # write the score and lives
        '''myfont = pygame.font.SysFont("monospace", 15)
        label = myfont.render("Score:" + str(self.score["score"]), 1, (255, 255, 0))
        self.screen.blit(label, (0, 0))
        label = myfont.render("Lives:" + str(self.score["lives"]), 1, (255, 255, 0))
        self.screen.blit(label, (100, 0))'''

    def update(self):
        self.prev = np.array(self.ball)
        self.ball += self.ball_vel
        self.ball_collision()
        self.ball_vel = np.clip(self.ball_vel, -4, 4)

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
            self.ball_vel = np.array([1, 2])
            self.ball = list(self.ball_initial)
            self.paddle = list(self.paddle_initial)
            self.score["lives"] -= 1
            return

        # paddle collision
        if ball[0] + radius < self.paddle[0] \
                or ball[0] - radius > self.paddle[0] + self.paddle[2] \
                or ball[1] + radius < self.paddle[1] \
                or ball[1] - radius > self.paddle[1] + self.paddle[3]:
            pass
        else:
            paddle = self.paddle
            if self.prev[1] + radius < paddle[1] \
                    or self.prev[1] - radius > paddle[1] + paddle[3]:
                if ball[0] < paddle[0] + 0.1 * paddle[2] or ball[0] > paddle[0] + 0.9 * self.paddle[2]:
                    ball_vel[0] *= -1
                    ball_vel[1] = 1
                    if ball_vel[0] < 0:
                        ball_vel[0] = -3
                    else:
                        ball_vel[0] = 3
                elif ball[0] < paddle[0] + 0.3 * paddle[2] or ball[0] > paddle[0] + 0.7 * paddle[2]:
                    ball_vel[0] *= -2
                    ball_vel[1] = 2
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
                ball_vel[0] += ball_vel[0] * 0.1
                ball_vel[1] += ball_vel[1] * 0.1
                brick[0] = self.w + 100
                self.score["score"] += 1

    def __init__(self, train=False):
        self.train = train
        self.w, self.h = 640, 480
        pw = 0.15 * self.w
        ph = 0.05 * self.h
        bw = self.w / 16
        bh = 16
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.w, self.h), 0, 32)
        pygame.display.set_caption('Breakout')
        pygame.mouse.set_visible(0)
        # game objects
        self.speed = 6
        self.radius = 6
        self.score = dict()
        self.entities_initial = [[i * bw, (j + 5) * bh, bw, bh] for i in range(16) for j in range(6)]
        self.ball = np.array([0, 0])
        self.ball_initial = (int(self.w / 2 - pw / 2), int(self.h / 2 - ph / 2))
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
        self.pixel_state = pygame.Surface((84, 84))
        self.surf = None

    def initialise(self):
        self.score["lives"] = 3
        self.score["score"] = 0
        self.start_ticks = pygame.time.get_ticks()
        self.paddle = list(self.paddle_initial)
        self.ball = np.array(self.ball_initial)
        self.ball_vel = np.array([random.randint(-1, 1), 2])
        bw = self.w / 16
        bh = 16
        self.entities = [[i * bw, (j + 5) * bh, bw, bh] for i in range(16) for j in range(6)]


if __name__ == "__main__":
    br = BreakoutEnv(train=True)
    br.train_agent()
