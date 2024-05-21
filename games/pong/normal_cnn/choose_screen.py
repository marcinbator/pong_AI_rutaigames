# import tkinter as tk
# import pyautogui
# from PIL import Image, ImageTk
#
# class RectangleSelector:
#     def __init__(self, root, background_image):
#         self.root = root
#         self.canvas = tk.Canvas(root, cursor="cross")
#         self.canvas.pack(fill=tk.BOTH, expand=True)
#         self.start_x = None
#         self.start_y = None
#         self.rect = None
#
#         # Display the screenshot as background
#         self.bg_image = background_image
#         self.tk_image = ImageTk.PhotoImage(self.bg_image)
#         self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
#
#         self.canvas.bind("<ButtonPress-1>", self.on_button_press)
#         self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
#         self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
#
#     def on_button_press(self, event):
#         self.start_x = event.x
#         self.start_y = event.y
#         self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='green')
#
#     def on_mouse_drag(self, event):
#         cur_x, cur_y = (event.x, event.y)
#         self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)
#
#     def on_button_release(self, event):
#         self.end_x = event.x
#         self.end_y = event.y
#         self.root.quit()
#
# def get_rectangle():
#     # Capture the entire screen using pyautogui
#     screenshot = pyautogui.screenshot()
#
#     root = tk.Tk()
#     root.attributes("-fullscreen", True)
#     app = RectangleSelector(root, screenshot)
#     root.mainloop()
#     root.destroy()
#     return app.start_x, app.start_y, app.end_x, app.end_y
#
# def choose_positions():
#     x1, y1, x2, y2 = get_rectangle()
#     print("Start Point: ", (x1, y1))
#     print("End Point: ", (x2, y2))
#
#     # Capture the selected area using PIL
#     # screenshot = pyautogui.screenshot(region=(x1, y1, x2 - x1, y2 - y1))
#     # screenshot.show()  # Display the selected area