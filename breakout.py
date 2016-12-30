import json
import pygame, sys, math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten
from keras.models import model_from_json

black = (0, 0, 0)
red = (255, 0, 0)
blue = (0, 0, 255)
white = (255, 255, 255)


class QLearning:
    def __init__(self, epsilon=0.1, frames=4, max_memory=4096, batch_size=128, num_actions=3, discount=0.9):
        self.epsilon = epsilon
        self.frames = frames
        self.max_memory = max_memory
        self.batch_size = batch_size
        self.model = None
        self.num_actions = num_actions
        self.memory = []
        self.discount = discount
        self.loss = 0
        self.conv_model()

    def conv_model(self):
        self.model = Sequential()
        self.model.add(Convolution2D(32, 8, 8, subsample=(4, 4), dim_ordering='th', input_shape=(self.frames, 84, 84),
                                     border_mode='valid'))
        self.model.add(Activation("relu"))
        self.model.add(Convolution2D(64, 4, 4, subsample=(2, 2), dim_ordering='th', input_shape=(32, 20, 20), border_mode='valid'))
        self.model.add(Activation("relu"))
        self.model.add(Convolution2D(64, 3, 3, dim_ordering='th', input_shape=(64, 9, 9), border_mode='valid'))
        self.model.add(Activation("relu"))
        self.model.add(Flatten())
        self.model.add(Dense(512, input_dim=3136))
        self.model.add(Activation("relu"))
        self.model.add(Dense(self.num_actions))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        # self.model.compile(loss='mse', optimizer='sgd')

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(0, self.num_actions, size=1)
        else:
            q = self.model.predict(state)
            action = np.argmax(q[0])
            print(q, action)
        return action

    def replay(self, data):
        if len(self.memory) + 1 < self.max_memory:
            self.memory.append(data)
        else:
            del self.memory[0]
            self.memory.append(data)

    def get_batch(self):
        memory_size = len(self.memory)
        batch_indices = np.random.randint(0, memory_size, self.batch_size)
        inputs = np.asarray([self.memory[i][0] for i in batch_indices])

        targets = np.zeros((self.batch_size, self.num_actions))
        for i, idx in enumerate(batch_indices):
            state, action, reward, new_state, game_over = self.memory[idx]
            targets[i] = self.model.predict(state)[0]
            q = np.max(self.model.predict(new_state)[0])
            if game_over:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount * q
        inputs = np.reshape(inputs, [self.batch_size, 4, 84, 84])
        return inputs, targets

    def train(self):
        print("train")
        for e in range(10):
            inputs, targets = self.get_batch()
            self.loss = self.model.train_on_batch(inputs, targets)


class BreakoutEnv:
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
        self.ball = [0, 0]
        self.ball_initial = (int(self.w / 2 - pw / 2), int(self.h / 2 - ph / 2))
        self.ball_vel = [0, 0]
        ball_surface = pygame.Surface((2 * self.radius, 2 * self.radius))
        ball_surface.fill(black)
        ball_surface.set_colorkey(black)
        pygame.draw.circle(ball_surface, (255, 255, 255), (self.radius, self.radius), self.radius)
        ball_surface.set_alpha(50)
        self.ball_surface = ball_surface
        self.paddle_initial = (self.w / 2 - pw / 2, self.h - ph, pw, ph)
        self.paddle = [0, 0, 0, 0]
        self.prev = [0, 0]
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
        self.ball = list(self.ball_initial)
        self.ball_vel = [1, 2]
        bw = self.w / 16
        bh = 16
        self.entities = [[i * bw, (j + 5) * bh, bw, bh] for i in range(16) for j in range(6)]

    def change(self):
        last_score = self.score["score"]
        last_lives = self.score["lives"]
        seconds = (pygame.time.get_ticks() - self.start_ticks) / 1000
        if seconds >= 1:
            self.start_ticks = pygame.time.get_ticks()
        return last_score, last_lives, seconds

    def observe(self, score, lives, seconds):
        reward = (self.score["score"] - score) * 9
        reward += (self.score["lives"] - lives) * 10
        if (self.score["lives"] - lives) == 0:
            reward += 1
        reward += seconds * 0.1
        game_over = False
        if self.score["score"] >= len(self.entities_initial):
            reward += 10
            game_over = True
        if self.score["lives"] <= 0:
            reward -= 300
            game_over = True
        return reward, game_over

    def get_frame(self):
        pygame.transform.scale(self.screen, (84, 84), self.pixel_state)
        arr = pygame.surfarray.pixels3d(self.pixel_state)
        avgs = [[(r * 0.298 + g * 0.587 + b * 0.114) for (r, g, b) in col] for col in arr]
        arr = np.array([[avg for avg in col] for col in avgs])
        arr2 = np.array([[[avg, avg, avg] for avg in col] for col in avgs])
        self.surf = pygame.surfarray.make_surface(arr2)
        return np.reshape(arr, [84, 84])

    def play(self):
        self.initialise()
        agent = QLearning(epsilon=0.0, frames=1)
        with open("model.json", "r") as jfile:
            agent.model = model_from_json(json.load(jfile))
        agent.model.load_weights("model.h5")
        agent.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        game_over = False
        s = self.get_state()
        while not game_over:
            action = agent.get_action(s)
            s_new = self.get_state(action)
            pygame.display.update()
            s = s_new

    def get_state(self, action=None):
        state = []
        for i in range(4):
            self.step(action)
            state.append(self.get_frame())
        state = np.asarray(state)
        return np.reshape(state, (1, 4, 84, 84))

    def start(self, epochs=1000):
        self.initialise()
        agent = QLearning(epsilon=0.2, frames=4)

        for e in range(epochs):
            game_over = False
            self.initialise()
            state = self.get_state()
            # agent.memory = []
            while not game_over:
                score, lives, seconds = self.change()
                action = agent.get_action(state)
                next_state = self.get_state(action)
                reward, game_over = self.observe(score, lives, seconds)
                agent.replay([state, action, reward, next_state, game_over])
                if self.surf is not None:
                    self.screen.blit(self.surf, (0, 0))
                pygame.display.update()
                state = next_state
            agent.train()
            agent.epsilon -= 0.01
            print('epoch {} loss {} memory {}'.format(e, agent.loss, len(agent.memory)))
            # Save trained model weights and architecture, this will be used by the visualization code
            agent.model.save_weights("model.h5", overwrite=True)
            with open("model.json", "w") as outfile:
                json.dump(agent.model.to_json(), outfile)

    def step(self, action=None):
        # msElapsed = self.clock.tick(144)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill(black)

        keys = pygame.key.get_pressed()
        if action is None:
            if keys[pygame.K_LEFT]:
                self.paddle[0] -= self.speed
                self.paddle[0] = max(0, self.paddle[0])
            if keys[pygame.K_RIGHT]:
                self.paddle[0] += self.speed
                self.paddle[0] = min(self.w - self.paddle[2], self.paddle[0])
        else:
            if action == 0:
                self.paddle[0] -= self.speed
                self.paddle[0] = max(0, self.paddle[0])
            if action == 2:
                self.paddle[0] += self.speed
                self.paddle[0] = min(self.w - self.paddle[2], self.paddle[0])

        # GAME OVER???
        if self.train is False:
            if self.score["lives"] <= 0:
                myfont2 = pygame.font.SysFont("monospace", 35, 1)
                label = myfont2.render("GAME OVER!", 1, (255, 0, 0))
                self.screen.blit(label, (self.w / 2 - label.get_width() / 2, self.h / 2 - label.get_height() / 2))

                myfont = pygame.font.SysFont("monospace", 15)
                label = myfont.render("Press RETURN to try again", 1, (255, 255, 0))
                self.screen.blit(label, (self.w / 2 - label.get_width() / 2, self.h / 2 - label.get_height() / 2 + 30))
                if keys[pygame.K_RETURN]:
                    self.initialise()
            else:
                self.update()
        else:
            self.update()

        pygame.draw.rect(self.screen, red, self.paddle, 0)

        for shape in self.entities:
            pygame.draw.rect(self.screen, blue, shape, 0)
            pygame.draw.rect(self.screen, black, shape, 1)

        new_ball = [int(i) for i in self.ball]
        magnitude = math.sqrt(self.ball_vel[0] * self.ball_vel[0] + self.ball_vel[1] * self.ball_vel[1])
        vel = [-v / magnitude for v in self.ball_vel]
        for i in range(int(np.abs(self.ball_vel[0]) + np.abs(self.ball_vel[1]))):
            prev_ball = (new_ball[0] + 2 * i * vel[0], new_ball[1] + 2 * i * vel[1])
            self.screen.blit(self.ball_surface, (prev_ball[0] - self.radius, prev_ball[1] - self.radius))
        pygame.draw.circle(self.screen, white, new_ball, self.radius)

        # write the score and lives
        myfont = pygame.font.SysFont("monospace", 15)
        label = myfont.render("Score:" + str(self.score["score"]), 1, (255, 255, 0))
        self.screen.blit(label, (0, 0))
        label = myfont.render("Lives:" + str(self.score["lives"]), 1, (255, 255, 0))
        self.screen.blit(label, (100, 0))

    def update(self):
        self.prev = [i for i in self.ball]
        self.ball[0] += self.ball_vel[0]
        self.ball[1] += self.ball_vel[1]
        self.ball_collision()
        if self.ball_vel[0] > 4:
            self.ball_vel[0] = 4
        if self.ball_vel[1] > 4:
            self.ball_vel[1] = 4
        if self.ball_vel[0] < -4:
            self.ball_vel[0] = -4
        if self.ball_vel[1] < -4:
            self.ball_vel[1] = -4

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
            self.ball_vel = [1, 2]
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


if __name__ == "__main__":
    br = BreakoutEnv(train=True)
    br.start()
