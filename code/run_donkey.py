import tensorflow as tf
import numpy as np
import gym
import gym_donkeycar
import cv2
import model

tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.InteractiveSession()
saver = tf.compat.v1.train.Saver()
saver.restore(sess, "save/model.ckpt")

# DonkeyCar environment
env = gym.make("donkey-warren-track-v0", conf={"cam_width": 800, "cam_height": 600})
obs = env.reset()

# Load steering wheel image
smoothed_angle = 0

try:
    while True:
        # preprocess observation
        frame = cv2.resize(obs, (200, 66)) / 255.0

        # steering prediction
        degrees = model.y.eval(feed_dict={model.x: [frame], model.keep_prob: 1.0})[0][0] * 180 / np.pi

        # smooth transitions
        smoothed_angle += 0.2 * pow(abs(degrees - smoothed_angle), 2.0 / 3.0) * (
            (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
        )

        # normalize and control simulator
        steering = np.clip(smoothed_angle / 25.0, -1, 1)
        throttle = 0.1
        action = np.array([steering, throttle])
        obs, reward, done, info = env.step(action)
        env.render()

        # visualization
        cv2.imshow("Camera Feed", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

        print(f"Predicted steering angle: {degrees:.2f} degrees")

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

        if done:
            obs = env.reset()

except KeyboardInterrupt:
    pass

env.close()
cv2.destroyAllWindows()
sess.close()
