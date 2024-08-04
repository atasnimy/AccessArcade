import streamlit as st
import cv2
import numpy as np
from nose_detection import NoseDetector
import hashlib
import random
import time
from pathlib import Path
from nose_detection2 import NoseDetector2


# Initialize nose detector
nose_detector = NoseDetector()


# Define the new maze (1: wall, 0: path)
new_maze = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])


# Define start and end points (row, column)
start_point = (1, 1)
end_point = (10, 9)


def draw_ball(frame, ball_pos):
    ball_radius = 10
    cv2.circle(frame, (int(ball_pos[0]), int(ball_pos[1])), ball_radius, (0, 0, 255), -1)  # Red ball


def draw_start_end(frame, start_pos, end_pos):
    start_radius = 15
    end_radius = 15
    cv2.circle(frame, (int(start_pos[0]), int(start_pos[1])), start_radius, (0, 255, 0), -1)  # Green for start
    cv2.circle(frame, (int(end_pos[0]), int(end_pos[1])), end_radius, (255, 0, 0), -1)  # Blue for end


def is_in_wall(position, maze, scale_x, scale_y):
    x, y = position
    row = int(y / scale_y)
    col = int(x / scale_x)
    if row < 0 or row >= maze.shape[0] or col < 0 or col >= maze.shape[1]:
        return True  # Out of bounds
    return maze[row, col] == 1  # Check if it's a wall


# In-memory user storage (for demonstration purposes)
user_db = {}


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def check_password(stored_password, provided_password):
    return stored_password == hash_password(provided_password)


def registration_page():
    st.title("Registration")
    st.write("Please enter your details to register.")
    username = st.text_input("Username", key="register_username")
    password = st.text_input("Password", type="password", key="register_password")
    if st.button("Register", key="register_button"):
        if username in user_db:
            st.error("Username already exists.")
        else:
            user_db[username] = hash_password(password)
            st.success("Registration successful!")
            st.session_state['authenticated'] = True
            st.session_state['current_page'] = "Dashboard"
            #st.experimental_rerun()






def login_page():
    st.title("Login")
    st.write("Please enter your credentials to login.")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login", key="login_button"):
        if username in "tester": # user_db and check_password(user_db[username], password):
            st.session_state['authenticated'] = True
            st.session_state['current_page'] = "Dashboard"
            st.success("Login successful!")
            #st.experimental_rerun()
        else:
            st.error("Invalid username or password.")


def show_login_register_page():
    st.title("Welcome")
    col1, col2 = st.columns(2)


    with col1:
        st.button("Login", on_click=lambda: st.session_state.update({'current_page': 'Login'}), key="show_login_button")


    with col2:
        st.button("Register", on_click=lambda: st.session_state.update({'current_page': 'Register'}), key="show_register_button")


def shared_menu():
    st.sidebar.title("Navigation")
    if st.session_state['authenticated']:
        st.sidebar.button("Dashboard", on_click=lambda: st.session_state.update({'current_page': 'Dashboard'}), key="dashboard_button")
        st.sidebar.button("Maze Escape", on_click=lambda: st.session_state.update({'current_page': 'Game'}), key="game_button")
        st.sidebar.button("Square Soar", on_click=lambda: st.session_state.update({'current_page': 'Flappy'}), key="flappy_button")
        st.sidebar.button("Profile", on_click=lambda: st.session_state.update({'current_page': 'Profile'}), key="profile_button")
        st.sidebar.button("Settings", on_click=lambda: st.session_state.update({'current_page': 'Settings'}), key="settings_button")
        st.sidebar.button("Logout", on_click=logout, key="logout_button")
    else:
        st.sidebar.button("Login", on_click=lambda: st.session_state.update({'current_page': 'Login'}), key="login_sidebar_button")
        st.sidebar.button("Register", on_click=lambda: st.session_state.update({'current_page': 'Register'}), key="register_sidebar_button")


def dashboard_page():
    st.title("Dashboard")
    st.write("Welcome to Accessible Games!\n\nOur goal is to make online games accessible to everyone.\n\nGames:\n- Maze Escape\n- Square Soar\n- Coming Soon")


def profile_page():
    st.title("Profile")
    st.write("Manage your profile information here.\n\nMore tools coming soon.")


def settings_page():
    st.title("Settings")
    st.write("Update your settings here.\n\nMore tools coming soon.")


def game_page():
    st.title('Maze Escape')
   
    # Create a placeholder for the video feed and game status
    stframe = st.empty()
    status_text = st.empty()
   
    # Video capture
    cap = cv2.VideoCapture(0)
   
    # Resize the maze to fit the 500x500 frame
    maze_size = (500, 500)
    maze_height, maze_width = new_maze.shape
    scale_x = maze_size[0] / maze_width
    scale_y = maze_size[1] / maze_height


    # Map maze coordinates to frame coordinates
    start_pos = (start_point[1] * scale_x + scale_x / 2, start_point[0] * scale_y + scale_y / 2)
    end_pos = (end_point[1] * scale_x + scale_x / 2, end_point[0] * scale_y + scale_y / 2)


    # Initialize ball position at the start position
    ball_pos = list(start_pos)
    prev_nose_pos = [0.5, 0.5]  # Initial nose position (center of the frame)


    exit_button_pressed = st.button('Exit', key='exit_button')


    if exit_button_pressed:
        st.write("Exiting game...")
        cap.release()
        cv2.destroyAllWindows()
        st.stop()  # Stop the Streamlit app


    game_over = False


    while not game_over:
        ret, frame = cap.read()
        if not ret:
            break


        nose_pos = nose_detector.detect_nose(frame)
        if nose_pos:
            # Normalize nose position
            nose_x, nose_y = nose_pos
            nose_x = min(max(nose_x, 0), 1)  # Clamp values to range [0, 1]
            nose_y = min(max(nose_y, 0), 1)


            # Calculate movement direction
            move_x = (nose_x - prev_nose_pos[0]) * 500
            move_y = (nose_y - prev_nose_pos[1]) * 500


            # Proposed new ball position
            new_ball_pos = [ball_pos[0] - move_x, ball_pos[1] + move_y]


            # Check if the new position is within a wall
            if not is_in_wall(new_ball_pos, new_maze, scale_x, scale_y):
                ball_pos = new_ball_pos


            # Update previous nose position
            prev_nose_pos = [nose_x, nose_y]


            # Check if the red ball has reached the blue ball
            if np.linalg.norm(np.array(ball_pos) - np.array(end_pos)) < 20:
                game_over = True


        # Create a blank frame for the maze
        maze_frame = np.zeros((maze_size[1], maze_size[0], 3), dtype=np.uint8)


        # Resize and map the maze to fit the frame
        resized_maze = cv2.resize(new_maze, (maze_size[0], maze_size[1]), interpolation=cv2.INTER_NEAREST)
        maze_frame[resized_maze == 1] = [0, 0, 0]  # Black for walls
        maze_frame[resized_maze == 0] = [255, 255, 255]  # White for paths


        # Draw start and end points
        draw_start_end(maze_frame, start_pos, end_pos)


        # Draw ball on maze frame
        draw_ball(maze_frame, ball_pos, (0, 0, 255))  # Red ball


        # Show frame in Streamlit
        stframe.image(cv2.cvtColor(maze_frame, cv2.COLOR_BGR2RGB), channels='RGB')


    # Display game over message
    status_text.write("Game Over! You reached the end.")
    st.write("Final Score: 100")  # Customize as needed
    st.balloons()


    cap.release()
    cv2.destroyAllWindows()


def flappy_page():
   
    st.title("Square Soar with Nose Detection")
    if 'bird_y' not in st.session_state:
        st.session_state.bird_y = SCREEN_HEIGHT // 2
    if 'bird_speed' not in st.session_state:
        st.session_state.bird_speed = 0
    if 'gravity' not in st.session_state:
        st.session_state.gravity = 0.5
    if 'score' not in st.session_state:
        st.session_state.score = 0
    if 'pipes' not in st.session_state:
        st.session_state.pipes = []
    if 'pipe_timer' not in st.session_state:
        st.session_state.pipe_timer = 0
    if 'last_pipe_x' not in st.session_state:
        st.session_state.last_pipe_x = SCREEN_WIDTH
    if 'passed_pipes' not in st.session_state:
        st.session_state.passed_pipes = set()
    if 'game_over' not in st.session_state:
        st.session_state.game_over = False
    if 'quit_game' not in st.session_state:
        st.session_state.quit_game = False


    nose_detector2 = NoseDetector2()


    if st.button('Start Game', key='start_game_button') and not st.session_state.game_over:
        st.session_state.quit_game = False
        st.session_state.game_over = False
        run_game(nose_detector2)


    if st.button('Reset', key='reset_button'):
        st.session_state.bird_y = SCREEN_HEIGHT // 2
        st.session_state.bird_speed = 0
        st.session_state.score = 0
        st.session_state.pipes = []
        st.session_state.game_over = False
        st.session_state.quit_game = False


def draw_start_end(frame, start, end):
    cv2.circle(frame, (int(start[0]), int(start[1])), 10, (0, 255, 0), -1)  # Green for start
    cv2.circle(frame, (int(end[0]), int(end[1])), 10, (255, 0, 0), -1)  # Blue for end


def draw_ball(frame, position, color):
    cv2.circle(frame, (int(position[0]), int(position[1])), 10, color, -1)


def is_in_wall(position, maze, scale_x, scale_y):
    maze_x = int(position[0] // scale_x)
    maze_y = int(position[1] // scale_y)
    return maze[maze_y, maze_x] == 1


def logout():
    st.session_state['authenticated'] = False
    st.session_state['current_page'] = "Login"
   # st.experimental_rerun()


def main():
    # Initialize session state if not present
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = "Login"


    shared_menu()


    if st.session_state['authenticated']:
        if st.session_state['current_page'] == "Dashboard":
            dashboard_page()
        elif st.session_state['current_page'] == "Profile":
            profile_page()
        elif st.session_state['current_page'] == "Settings":
            settings_page()
        elif st.session_state['current_page'] == "Game":
            game_page()
        elif st.session_state['current_page'] == "Flappy":
            flappy_page()
    else:
        if st.session_state['current_page'] == "Login":
            login_page()
        elif st.session_state['current_page'] == "Register":
            registration_page()
        else:
            show_login_register_page()


#flappy code


# Constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
BIRD_WIDTH = 40
BIRD_HEIGHT = 40
PIPE_WIDTH = 60
PIPE_GAP = 150
PIPE_SPEED = 8
FRAME_RATE = 30
PIPE_INTERVAL = 300


def add_pipe():
    y = random.randint(100, SCREEN_HEIGHT - PIPE_GAP - 100)
    # Use a unique id for each pipe
    pipe_id = len(st.session_state.pipes)
    st.session_state.pipes.append({'x': SCREEN_WIDTH, 'y': y, 'id': pipe_id})


def move_pipes():
    new_pipes = []
    for pipe in st.session_state.pipes:
        pipe['x'] -= PIPE_SPEED
        if pipe['x'] + PIPE_WIDTH > 0:
            new_pipes.append(pipe)
    st.session_state.pipes = new_pipes


def draw_shadow(frame, obj_x, obj_y, width, height):
    shadow_color = (50, 50, 50)
    cv2.rectangle(frame, (obj_x + 5, obj_y + 5), (obj_x + width + 5, obj_y + height + 5), shadow_color, -1)


def draw_bird(frame):
    bird_color = (255, 255, 0)
    draw_shadow(frame, 50, int(st.session_state.bird_y), BIRD_WIDTH, BIRD_HEIGHT)
    cv2.rectangle(frame, (50, int(st.session_state.bird_y)), (50 + BIRD_WIDTH, int(st.session_state.bird_y) + BIRD_HEIGHT), bird_color, -1)


def draw_pipes(frame):
    pipe_color = (34, 139, 34)
    for pipe in st.session_state.pipes:
        draw_shadow(frame, pipe['x'], 0, PIPE_WIDTH, pipe['y'])
        cv2.rectangle(frame, (pipe['x'], 0), (pipe['x'] + PIPE_WIDTH, pipe['y']), pipe_color, -1)
        draw_shadow(frame, pipe['x'], pipe['y'] + PIPE_GAP, PIPE_WIDTH, SCREEN_HEIGHT - (pipe['y'] + PIPE_GAP))
        cv2.rectangle(frame, (pipe['x'], pipe['y'] + PIPE_GAP), (pipe['x'] + PIPE_WIDTH, SCREEN_HEIGHT), pipe_color, -1)


def draw_background(frame):
    for y in range(SCREEN_HEIGHT):
        color = (173, 216, 230)
        gradient = int(255 - (255 * (y / SCREEN_HEIGHT)))
        frame[y, :] = (color[2], color[1], color[0])


def check_collision():
    for pipe in st.session_state.pipes:
        if (50 + BIRD_WIDTH > pipe['x'] and 50 < pipe['x'] + PIPE_WIDTH and
            (st.session_state.bird_y < pipe['y'] or st.session_state.bird_y + BIRD_HEIGHT > pipe['y'] + PIPE_GAP)):
            return True
    return False


def update_score():
    for pipe in st.session_state.pipes:
        # Check if the pipe has moved past the bird's x position and hasn't been scored yet
        if pipe['x'] + PIPE_WIDTH < 50 and pipe['id'] not in st.session_state.passed_pipes:
            st.session_state.score += 1
            st.session_state.passed_pipes.add(pipe['id'])


def display_score(score):
    st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>Score: {score}</h1>", unsafe_allow_html=True)


def run_game(nose_detector2):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error opening video stream or file")
        return


    st.session_state.pipes = []
    st.session_state.score = 0
    st.session_state.bird_y = SCREEN_HEIGHT // 2
    st.session_state.bird_speed = 0
    st.session_state.game_over = False
    st.session_state.passed_pipes = set()
    st.session_state.pipe_timer = 0
    st.session_state.last_pipe_x = SCREEN_WIDTH


    game_placeholder = st.empty()
    score_placeholder = st.empty()


    while cap.isOpened() and not st.session_state.quit_game:
        ret, frame = cap.read()
        if not ret:
            break


        nose_x, nose_y = nose_detector2.detect_nose(frame)
        if nose_y is not None:
            st.session_state.bird_y = int(nose_y / 480 * SCREEN_HEIGHT)


        st.session_state.bird_speed += st.session_state.gravity
        st.session_state.bird_y += st.session_state.bird_speed


        st.session_state.pipe_timer += PIPE_SPEED
        if st.session_state.pipe_timer >= PIPE_INTERVAL:
            add_pipe()
            st.session_state.pipe_timer = 0


        move_pipes()
        update_score()


        game_frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        draw_background(game_frame)


        draw_bird(game_frame)
        draw_pipes(game_frame)


        game_frame_rgb = cv2.cvtColor(game_frame, cv2.COLOR_BGR2RGB)


        game_placeholder.image(game_frame_rgb, channels="RGB", use_column_width=True)
        score_placeholder.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>Score: {st.session_state.score}</h1>", unsafe_allow_html=True)


        if check_collision() or st.session_state.bird_y >= SCREEN_HEIGHT or st.session_state.bird_y <= 0:
            st.write("Game Over!")
            st.session_state.game_over = True
            break


        time.sleep(1 / FRAME_RATE)


    cap.release()
    cv2.destroyAllWindows()


# Main Application


#if st.session_state.logged_in:
 #   st.sidebar.title("Navigation")
  #  app_mode = st.sidebar.selectbox("Choose the app mode", ["Game", "Profile", "Settings"])


   # if app_mode == "Game":


 #   elif app_mode == "Profile":
  #      st.title("Profile Page")
   #     st.write(f"Welcome, {st.session_state.username}!")


#    elif app_mode == "Settings":
 #       st.title("Settings Page")
  #      st.write("Settings content goes here.")


   # if st.sidebar.button('Logout'):
    #    st.session_state.logged_in = False
     #   st.session_state.username = None


if __name__ == "__main__":
    main()


