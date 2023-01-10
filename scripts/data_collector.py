"""Data collector for the RuneLite client"""
import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime

import cv2
import cvzone
import keyboard
import mouse
import mss
import numpy as np
import pyautogui
from azure.storage.blob import BlobServiceClient

# Set global variables
VERSION = "1.0"
STORAGE_ACCOUNT_NAME = "rstrainingdata"
STORAGE_ACCOUNT_KEY = "SQUzMlTxJOkaaBddPYOAxPiy7n7y9lGCIOw1ZSGIVQwfCQpbnbCv2RQcl1TmzGDbMogdBPfHF5l9+AStcbYWHA=="
CONTAINER_NAME = "data"
FRAME_RATE = 15
SLEEP_TIME = 1.0 / FRAME_RATE


def create_run_dir(path: str):
    """Create the run directory"""
    if not os.path.exists(path):
        os.makedirs(path)


def get_runelite_window() -> pyautogui.Window or None:
    """Get the window object for the RuneLite client

    Returns:
        pyautogui.Window: The RuneLite client window
    """
    # Get a list of all the windows on the screen
    windows = pyautogui.getAllWindows()
    # Search for the window with the title "RuneLite"
    for win in windows:
        if win.title.startswith("RuneLite"):
            return win
    return None


def get_mouse_data(window) -> tuple:
    """Get the mouse position relative to the RuneLite client window

    Returns:
        tuple: The mouse position
    """
    # Get the mouse position
    mouse_x, mouse_y = mouse.get_position()
    # Check if the mouse is inside the RuneLite client window
    if (
        window.left <= mouse_x <= window.right
        and window.top <= mouse_y <= window.bottom
    ):
        mouse_inside = True
    else:
        mouse_inside = False
    # Calculate the mouse position relative to the RuneLite client window
    mouse_x -= window.left
    mouse_y -= window.top
    return (mouse_x, mouse_y), mouse_inside


def get_keyboard_input() -> list:
    """Get the keyboard input

    Returns:
        list: The keyboard input
    """
    # Get the keyboard input
    keyboard_event = keyboard.stop_recording()
    # Convert the keyboard input to a list of keys
    keys = []
    for event in keyboard_event:
        if event.event_type == "down":
            keys.append(event.name)
    return keys


def get_mouse_input(mouse_data: tuple) -> dict:
    """Get the mouse input

    Returns:
        dict: The mouse input
    """
    # Get the mouse position and check if the mouse is inside the window
    mouse_pos, mouse_inside = mouse_data
    # Get the mouse input
    mouse_data = {
        "mouse_pos": mouse_pos,
        "left_click": mouse.is_pressed("left"),
        "right_click": mouse.is_pressed("right"),
        "middle_click": mouse.is_pressed("middle"),
        "inside": mouse_inside,
    }
    return mouse_data


def get_frame(window, screenshot) -> np.ndarray:
    """Get a frame of the RuneLite client window

    Returns:
        np.ndarray: The frame
    """
    # get x, y, width, height of the window in case it has been moved
    x, y, width, height = window.left, window.top, window.width, window.height
    # Capture a frame of the window
    frame = screenshot.grab(
        {"left": x, "top": y, "width": width, "height": height}
    )
    # Convert the frame to a numpy array
    frame = np.array(frame)
    # Convert the frame from 4 channels to 3 channels
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def draw_cursor(
    frame: np.ndarray, mouse_info: dict, cursor: np.ndarray
) -> np.ndarray:
    """Draw the cursor on the frame

    Args:
        frame (np.ndarray): The frame
        mouse_info (dict): A dictionary containing information about the mouse position and whether the mouse is inside the RuneLite client window
    """

    # Get the mouse position
    mouse_pos = mouse_info["mouse_pos"]
    # Check if the mouse is inside the RuneLite client window
    if mouse_info["inside"]:
        # Add padding to the frame
        frame = cv2.copyMakeBorder(
            frame, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        # Calculate the mouse position relative to the padded frame
        mouse_pos = (mouse_pos[0] + 100, mouse_pos[1] + 100)
        # Draw the cursor on the frame
        frame = cvzone.overlayPNG(frame, cursor, pos=mouse_pos)
        # Remove the padding from the frame
        frame = frame[100:-100, 100:-100]
    return frame


def save_frame(frame: np.ndarray, path: str, count: int):
    """Save a frame to the run directory

    Args:
        frame (np.ndarray): The frame
        count (int): The frame count
    """
    cv2.imwrite(os.path.join(path, f"frame{count}.jpeg"), frame)


async def save_blob(
    frame: np.ndarray,
    run_uuid: str,
    count: int,
    blob_service: BlobServiceClient,
):
    """Save a frame to Azure Blob Storage

    Args:
        frame (np.ndarray): The frame
        count (int): The frame count
    """
    try:
        # Convert the frame to a bytes-like object
        _, frame_bytes = cv2.imencode(".jpeg", frame)
        blob_service.get_blob_client(
            CONTAINER_NAME, f"{run_uuid}/frame{count}.jpeg"
        ).upload_blob(frame_bytes.tobytes())
    except Exception as ex:
        # Log the error message
        print("Failed to save frame to Azure Blob Storage")
        print(ex)


def save_json(data: dict, path: str):
    """Save the data to a json file

    Args:
        data (dict): The data
    """
    with open(os.path.join(path, "data.json"), "w", encoding="utf-8") as file:
        json.dump(data, file)


def save_json_azure(data: dict, run_uuid: str, blob_service: BlobServiceClient):
    """Save the data to Azure Blob Storage

    Args:
        data (dict): The data
    """
    try:
        # Convert the data to a bytes-like object
        data_bytes = json.dumps(data).encode("utf-8")
        blob_service.get_blob_client(
            CONTAINER_NAME, f"{run_uuid}/data.json"
        ).upload_blob(data_bytes)
    except Exception as ex:
        # Log the error message
        print("Failed to save data to Azure Blob Storage")
        print(ex)


def get_player(title: str) -> str:
    return title.split(" - ")[1]


def create_run_id_for(player: str) -> str:
    """Create a unique run ID for the player

    Args:
        player (str): The player name

    Returns:
        str: The run ID (e.g. <player>-<session-id(12 characters)>-<date>-<time>)
    """
    return f"{player}-{uuid.uuid4().hex[:12]}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"


async def main():
    """The main function"""

    # Check if the RuneLite client window was found
    runelite_window = get_runelite_window()
    if runelite_window is None:
        print("Could not find RuneLite client window")
        sys.exit()

    # Get the player name
    player = get_player(runelite_window.title)
    # Create a unique run ID for the player
    run_uuid = create_run_id_for(player)
    # Create a directory for the run
    run_directory = os.path.join(CONTAINER_NAME, run_uuid)

    if SAVE_LOCALLY:
        create_run_dir(run_directory)

    count = 0
    action_data = []

    # Start the timer
    start_time = time.time()

    # Get the cursor image from the assets folder
    cursor = cv2.imread("assets/cursor-rs3-gold.png", cv2.IMREAD_UNCHANGED)

    screenshot = mss.mss()

    # Create a new instance of the BlobServiceClient class
    blob_service = BlobServiceClient(
        f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
        credential=STORAGE_ACCOUNT_KEY,
    )

    # Create the container if it doesn't already exist
    if not blob_service.get_container_client(CONTAINER_NAME).exists():
        blob_service.create_container(CONTAINER_NAME)

    # Start the main loop
    while True:
        # Start frame timer
        frame_time = time.perf_counter()
        # Get the mouse input
        mouse_data = get_mouse_data(runelite_window)
        mouse_info = get_mouse_input(mouse_data)

        # Get a frame of the RuneLite client window
        frame = get_frame(runelite_window, screenshot)

        # Draw the cursor on the frame
        frame = draw_cursor(frame, mouse_info, cursor)

        # Save the frame
        if SAVE_LOCALLY:
            save_frame(frame, run_directory, count)
        else:
            # Save the frame to Azure Blob Storage
            await save_blob(frame, run_uuid, count, blob_service)

        # Get the action data
        action_data.append(
            {
                "time": time.time() - start_time,
                "mouse_pos": mouse_info["mouse_pos"],
                "left_click": mouse_info["left_click"],
                "right_click": mouse_info["right_click"],
                "middle_click": mouse_info["middle_click"],
                "inside": mouse_info["inside"],
            }
        )

        # Increment the frame count
        count += 1

        # calculate the fps and print it in place of the previous fps
        fps = 1 / (time.perf_counter() - frame_time)
        print(f"FPS: {fps:.2f}", end="\r")

        # Calculate the elapsed time since the last frame
        elapsed_time = time.perf_counter() - frame_time
        time_to_sleep = max(0, SLEEP_TIME - elapsed_time)
        time.sleep(time_to_sleep)

        # If the user presses "Shift + Esc", exit the program
        if keyboard.is_pressed("shift+esc"):
            break

    # Save the data
    if SAVE_LOCALLY:
        save_json(action_data, run_directory)
    else:
        save_json_azure(action_data, run_uuid, blob_service)
    print("Done")
    # Close the screenshot object
    screenshot.close()
    # Exit the program
    sys.exit()


if __name__ == "__main__":
    # get arguments
    parser = argparse.ArgumentParser()
    # If --local is passed, save data locally
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    SAVE_LOCALLY = args.local  # Save data locally if --local is passed
    # Run the main function
    asyncio.run(main())
