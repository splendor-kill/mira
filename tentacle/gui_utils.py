import time
import pyautogui
import subprocess
import re


def xwininfo_output(name):
    output = subprocess.run(["xwininfo", "-name", name], stdout=subprocess.PIPE).stdout
    output = output.decode('ascii')
    return output


def extract_dims(output):
    mo = re.search('  Absolute upper-left X:  (\d+)', output)
    x = int(mo.group(1))
    mo = re.search('  Absolute upper-left Y:  (\d+)', output)
    y = int(mo.group(1))
    mo = re.search('  Width: (\d+)', output)
    w = int(mo.group(1))
    mo = re.search('  Height: (\d+)', output)
    h = int(mo.group(1))
    return x, y, w, h


def get_xwin_dims(name):
    output = xwininfo_output(name)
    return extract_dims(output)


def capture_window(x, y, w, h, h_nc):
    im = pyautogui.screenshot(region=(x, y + h_nc, w, h - h_nc))
    return im



if __name__ == '__main__':
    time.sleep(5)
    x, y, w, h = get_xwin_dims('Freecell')
#     x, y, w, h = get_xwin_dims('Sudoku')
    print(x, y, w, h)

# pyautogui.mouseDown(1360, 670)
# pyautogui.moveTo(1360, 670)
# pyautogui.dragTo(220, 330, duration=0.5)
# pyautogui.mouseUp(220, 330, duration=0.5)

# where = pyautogui.locateOnScreen('/home/splendor/Pictures/freecell-new.png')
# print(where)
# where = pyautogui.center(where)
# print(where)
# pyautogui.click(where)


# pyautogui.click()
# distance = 200
# while distance > 0:
#     pyautogui.dragRel(distance, 0, duration=0.2)
#     distance = distance - 5
#     pyautogui.dragRel(0, distance, duration=0.2)
#     pyautogui.dragRel(-distance, 0, duration=0.2)
#     distance = distance - 5
#     pyautogui.dragRel(0, -distance, duration=0.2)


# print('Press Ctrl-C to quit.')
# try:
#     while True:
#         x, y = pyautogui.position()
#         positionStr = 'X: ' + str(x).rjust(4) + ' Y: ' + str(y).rjust(4)
#         print(positionStr, end='')
#         print('\b' * len(positionStr), end='', flush=True)
# except KeyboardInterrupt:
#     print('\nDone.')
