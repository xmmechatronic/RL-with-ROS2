import rclpy #ros2 çalıştırmak için rclpy kütüphanesine ihtiyacımız var
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

import numpy as np
import random
import time
import matplotlib.pyplot as plt


class QLearningNode(Node):

    def __init__(self):
        super().__init__('q_learning_node')

        # Publishers / Subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        # Reset service
        self.reset_client = self.create_client(Empty, '/reset_simulation')
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for reset_simulation service...')

        # Timer
        self.timer = self.create_timer(0.2, self.step)

        # Q-learning parameters
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.9
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        # Actions: 0=forward, 1=left, 2=right
        self.actions = [0, 1, 2]

        self.q_table = {}

        # State tracking
        self.state = None
        self.prev_state = None
        self.prev_action = None

        # Episode tracking
        self.episode = 0
        self.step_count = 0
        self.episode_steps = []  # <-- grafik için

        self.get_logger().info("Q-Learning Node Started")

    # ------------------------------------------------
    # LIDAR CALLBACK → STATE
    # ------------------------------------------------
    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges[ranges == 0.0] = 10.0

        front = np.min(ranges[0:20])
        left  = np.min(ranges[60:100])
        right = np.min(ranges[260:300])

        self.state = (
            round(front, 1),
            round(left, 1),
            round(right, 1)
        )

    # ------------------------------------------------
    # MAIN STEP LOOP
    # ------------------------------------------------
    def step(self):
        if self.state is None:
            return

        self.step_count += 1

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            action = self.get_best_action(self.state)

        self.take_action(action)

        reward, done = self.get_reward(self.state)

        if self.prev_state is not None:
            self.update_q(self.prev_state, self.prev_action, reward, self.state)

        if done:
            self.reset_episode()
            return

        self.prev_state = self.state
        self.prev_action = action

    # ------------------------------------------------
    # ACTION EXECUTION
    # ------------------------------------------------
    def take_action(self, action):
        cmd = Twist()

        if action == 0:
            cmd.linear.x = 0.2
        elif action == 1:
            cmd.angular.z = 0.6
        elif action == 2:
            cmd.angular.z = -0.6

        self.cmd_pub.publish(cmd)

    # ------------------------------------------------
    # REWARD FUNCTION
    # ------------------------------------------------
    def get_reward(self, state):
        front, left, right = state

        if front < 0.25:
            return -10.0, True
        elif front > 1.0:
            return 2.0, False
        else:
            return -0.1, False

    # ------------------------------------------------
    # Q TABLE UPDATE
    # ------------------------------------------------
    def update_q(self, state, action, reward, next_state):
        old_q = self.q_table.get((state, action), 0.0)

        next_max = max(
            self.q_table.get((next_state, a), 0.0)
            for a in self.actions
        )

        new_q = old_q + self.alpha * (reward + self.gamma * next_max - old_q)
        self.q_table[(state, action)] = new_q

    def get_best_action(self, state):
        q_values = [
            self.q_table.get((state, a), 0.0)
            for a in self.actions
        ]
        return self.actions[int(np.argmax(q_values))]

    # ------------------------------------------------
    # EPISODE RESET
    # ------------------------------------------------
    def reset_episode(self):
        self.episode += 1
        self.episode_steps.append(self.step_count)

        self.get_logger().info(
            f"Episode: {self.episode} | Steps: {self.step_count} | Epsilon: {round(self.epsilon, 3)}"
        )

        self.step_count = 0

        self.epsilon = max(
            self.epsilon * self.epsilon_decay,
            self.epsilon_min
        )

        req = Empty.Request()
        self.reset_client.call_async(req)

        self.prev_state = None
        self.prev_action = None

        time.sleep(1.0)

    # ------------------------------------------------
    # SHUTDOWN → PLOT
    # ------------------------------------------------
    def plot_results(self):
        if len(self.episode_steps) == 0:
            return

        plt.figure()
        plt.plot(self.episode_steps)
        plt.xlabel("Episode")
        plt.ylabel("Steps (time before collision)")
        plt.title("Q-Learning Performance")
        plt.grid(True)
        plt.show()


def main():
    rclpy.init()
    node = QLearningNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.plot_results()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
