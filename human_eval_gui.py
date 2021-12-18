from tkinter import *
from enum import Enum
import numpy as np
import json
import os


DATA_PATH = "cached_data"

class View(Enum):
    SA = 'sa'
    SALE = 'sale'
    LA = 'la'
    LALE = 'lale'


class State(Enum):
    LOGIN = 0  # a new user starts the scoring
    SCORE = 1  # the images are shown and the user can evaluate the curves
    SCORED = 2  # the user has choosen an evaluation score
    LAST_PAGE = 3  # when the user finished all the scoring


class Action(Enum):
    NEUTRAL = 0
    PAGING = 1  # go to the next page
    SCORING1 = 2
    SCORING2 = 3
    SCORING3 = 4
    SCORING4 = 5
    SCORING5 = 6
    SCORING6 = 7
    SCORING7 = 8
    SCORING8 = 9
    SCORING9 = 10
    SCORING10 = 11


pathologies = {
    1: "HCM",
    2: "Infarktus",
    3: "DCM",
    4: "Myocarditis",
    5: "EMF",
    6: "Amyloidosis",
    7: "Aortastenosis",
    8: "Tako Tsubo",
    9: "Fabry",
    10: "Normal"
}


class Page:

    def __init__(self):
        self.page_num = -1  # no images when starts up
        self.pages = 0  # number of pages
        self.name = "demo"
        self.paths = None
        self.text = None
        self.num_images = dict()
        self.chosen_image = (0, 0)
        self.scoring = None
        self.perm = Permutator()
    
    def get_perm_page_num(self):
        return self.perm.get(self.page_num)


class Permutator:

    def __init__(self):
        self.path = os.path.join(DATA_PATH, "scores", "auto_perm.json")
        if not os.path.exists(self.path):
            self.__generate_permutation()
        self.permutation = self.__read_permutation()
    
    def __generate_permutation(self):
        folder_numbers = [int(fn) for fn in os.listdir(DATA_PATH) if fn != "scores"]
        np.random.shuffle(folder_numbers)
        mapping = {idx: fn for idx, fn in enumerate(folder_numbers, 0)}
        with open(self.path, 'wt') as js:
            json.dump(mapping, js)
    
    def __read_permutation(self):
        mapping = None
        with open(self.path, 'rt') as js:
            mapping = json.load(js)
        return mapping
    
    def get(self, page_idx):
        return self.permutation[str(page_idx)]


class StateManager:
    
    def __init__(self, cntmg, ptncmg):
        self.cntmg = cntmg
        self.ptncmg = ptncmg
        self.state = State.LOGIN
    
    def switch_table(self):
        nxt_paging = (State.LAST_PAGE if self.ptncmg.next_is_last() else State.SCORE)
        st = {
            State.LOGIN: {Action.NEUTRAL: State.LOGIN, Action.PAGING: nxt_paging},
            State.SCORE: {
                Action.NEUTRAL: State.SCORE,
                Action.PAGING: State.SCORE, 
                Action.SCORING1: State.SCORED, 
                Action.SCORING2: State.SCORED, 
                Action.SCORING3: State.SCORED, 
                Action.SCORING4: State.SCORED, 
                Action.SCORING5: State.SCORED,
                Action.SCORING6: State.SCORED,
                Action.SCORING7: State.SCORED,
                Action.SCORING8: State.SCORED,
                Action.SCORING9: State.SCORED,
                Action.SCORING10: State.SCORED
            },
            State.SCORED: {
                Action.NEUTRAL: State.SCORED,
                Action.PAGING: nxt_paging, 
                Action.SCORING1: State.SCORED, 
                Action.SCORING2: State.SCORED, 
                Action.SCORING3: State.SCORED, 
                Action.SCORING4: State.SCORED, 
                Action.SCORING5: State.SCORED,
                Action.SCORING6: State.SCORED,
                Action.SCORING7: State.SCORED,
                Action.SCORING8: State.SCORED,
                Action.SCORING9: State.SCORED,
                Action.SCORING10: State.SCORED
            },
            State.LAST_PAGE: {
                Action.NEUTRAL: State.LAST_PAGE
            }
        }
        return st
    
    def transition(self, action, w, h):
        self.ptncmg.transition(self.state, action)
        self.state = self.switch_table()[self.state][action]
        self.cntmg.transition(self.state, action, w, h)  # show the next page


class ContentManager:

    def __init__(self, canvas, widgets, page):
        self.canvas = canvas
        self.next_button = widgets[0]
        self.login_entry = widgets[1]
        self.login_label = widgets[2]
        self.scoring1_btn = widgets[3]
        self.scoring2_btn = widgets[4]
        self.scoring3_btn = widgets[5]
        self.scoring4_btn = widgets[6]
        self.scoring5_btn = widgets[7]
        self.scoring6_btn = widgets[8]
        self.scoring7_btn = widgets[9]
        self.scoring8_btn = widgets[10]
        self.scoring9_btn = widgets[11]
        self.scoring10_btn = widgets[12]
        self.selector_sa_btn = widgets[13]
        self.selector_sale_btn = widgets[14]
        self.selector_la_btn = widgets[15]
        self.selector_lale_btn = widgets[16]
        self.anamnezis = widgets[17]

        self.page = page
        self.view = View.SA.value
        self.w, self.h = self.default_sizes()
        self.tile_size = 0
        self.tile_positions = 0
        self.focused_size = 0
        self.focused_position = 0
        self.num_images = 0
        self.num_horizontal = 0
        self.num_vertical = 0
    
    @staticmethod
    def default_sizes():
        return 400, 300  # default width and height
    
    def transition(self, state, action, w, h):
        self.canvas.delete("all")
        self.w, self.h = w, h
        if state == State.LOGIN:
            self.login_page()
        elif state in [State.SCORE, State.SCORED]:
            action_id = int(action.value)
            if action == Action.PAGING:
                self.page.scoring = None
                self.page.chosen_image = (0, 0)
                self.view = View.SA.value
            elif 1 < action_id < 12:
                self.page.scoring = pathologies[action_id - 1]
            self.scoring_page()
        else:
            self.last_page()
    
    def login_page(self):
        self.login_entry.place(x=int(self.w * 0.4), y=(self.h * 0.5), width=self.w // 5)
        self.login_label.place(x=int(self.w * 0.4 - 70), y=(self.h * 0.5))
        # listing already existing names
        names = [fn[:-4] for fn in os.listdir(os.path.join(DATA_PATH, 'scores')) if fn.endswith('txt')]
        names_as_text = ' \n'.join(names)
        self.canvas.create_text((int(self.w * 0.05), int(self.h * 0.1)), anchor=W, text="Existing names: \n{}".format(names_as_text))
        self._resize_button()
    
    def scoring_page(self):

        def login_widgets():
            if self.login_entry is not None:
                self.page.name = str(self.login_entry.get())
                self.page.chosen_image = (0, 0)
                self.login_entry.destroy()
                self.login_entry = None
            if self.login_label is not None:
                self.login_label.destroy()
                self.login_label = None
        
        def widget_sizes():
            if not(self.view in self.page.num_images):
                self.num_images = 0
                return
            # size of the tiled images
            num_images = self.page.num_images[self.view]
            num_horizontal = 25
            num_vertical = int(num_images / num_horizontal) + 1
            height = int(self.h * 0.5) - 5
            width = int(self.w * 0.6)
            tile_image_height = (height - (2 * 5 + (num_vertical - 1) * 3)) / num_vertical
            tile_image_width = (width - (2 * 5 + (num_horizontal - 1) * 3)) / num_horizontal
            tile_size = min(tile_image_height, tile_image_width)
        
            # position of tiled images
            tile_positions = np.zeros((num_vertical, num_horizontal, 2))
            x = self.w * 0.4 
            y = self.h * 0.05
            for vrt in range(num_vertical):
                for hrn in range(num_horizontal):
                    x_new = int(x + hrn * (tile_size + 3))
                    y_new = int(y + vrt * (tile_size + 3))
                    tile_positions[vrt, hrn] = np.array([x_new, y_new])
        
            # size and position of huge image
            focused_size = int(self.h / 3.0 * 2)
            focused_position = (10 + (focused_size // 2),  10 + focused_size // 2)
        
            # save results
            self.num_images = num_images
            self.num_horizontal = num_horizontal
            self.num_vertical = num_vertical
            self.tile_size = tile_size
            self.tile_positions = tile_positions
            self.focused_size = focused_size
            self.focused_position = focused_position
        
        # drawer functions (display images and labels)
        def draw_squares():
            if self.num_images == 0:
                self.canvas.create_text((int(self.w * 0.5), int(self.h * 0.15)), anchor=W, text="Not available")
                return
            for i in range(self.num_vertical):
                for j in range(self.num_horizontal):
                    if (i * self.num_horizontal + j) < self.num_images:
                        color = ('green' if self.page.chosen_image == (i, j) else 'gray')
                        xy = self.tile_positions[i, j]
                        self.canvas.create_rectangle((xy[0], xy[1], int(xy[0] + self.tile_size), int(xy[1] + self.tile_size)), fill=color)
    
        def show_image():
            if self.num_images == 0:
                return
            i, j = self.page.chosen_image
            idx = i * 25 + j
            path = self.page.paths[self.view][idx]
            self.focused_img = PhotoImage(file=path, cnf={'width':self.focused_size, 'height':self.focused_size}, master=self.canvas)
            self.canvas.create_image(self.focused_position[0], self.focused_position[1], image=self.focused_img)
        
        def draw_splitting_line():
            xy_horizontal = (self.canvas.canvasx(int(self.w * 0.37)), self.canvas.canvasy(int(self.h * 0.60) - 2), self.canvas.canvasx(self.w), self.canvas.canvasy(int(self.h * 0.60) + 2))
            self.canvas.create_rectangle(xy_horizontal, fill="blue", outline="blue")
            xy_vertical = (self.canvas.canvasx(int(self.w * 0.37)), self.canvas.canvasy(int(self.h * 0.60) - 2), self.canvas.canvasx(int(self.w * 0.37)+4), self.canvas.canvasy(int(self.h)))
            self.canvas.create_rectangle(xy_vertical, fill="blue", outline="blue")
        
        def name_page_label():
            self.canvas.create_text((int(self.w * 0.03), int(self.h * 0.70)), anchor=W, text="You are: {}".format(self.page.name))
            self.canvas.create_text((int(self.w * 0.03), int(self.h * 0.72)), anchor=W, text="Pg: {} / {}".format(self.page.page_num + 1, self.page.pages))
        
        def scoring_buttons():
            self.scoring1_btn.place(x=int(self.w * 0.39), y=(self.h * 0.70), width=self.w // 20, height=self.h // 20)
            self.scoring2_btn.place(x=int(self.w * 0.45), y=(self.h * 0.70), width=self.w // 20, height=self.h // 20)
            self.scoring3_btn.place(x=int(self.w * 0.51), y=(self.h * 0.70), width=self.w // 20, height=self.h // 20)
            self.scoring4_btn.place(x=int(self.w * 0.57), y=(self.h * 0.70), width=self.w // 20, height=self.h // 20)
            self.scoring5_btn.place(x=int(self.w * 0.63), y=(self.h * 0.70), width=self.w // 20, height=self.h // 20)
            self.scoring6_btn.place(x=int(self.w * 0.69), y=(self.h * 0.70), width=self.w // 20, height=self.h // 20)
            self.scoring7_btn.place(x=int(self.w * 0.75), y=(self.h * 0.70), width=self.w // 20, height=self.h // 20)
            self.scoring8_btn.place(x=int(self.w * 0.81), y=(self.h * 0.70), width=self.w // 20, height=self.h // 20)
            self.scoring9_btn.place(x=int(self.w * 0.87), y=(self.h * 0.70), width=self.w // 20, height=self.h // 20)
            self.scoring10_btn.place(x=int(self.w * 0.93), y=(self.h * 0.70), width=self.w // 20, height=self.h // 20)
            pathology = "Not answered yet." if self.page.scoring is None else str(self.page.scoring)
            self.canvas.create_text((int(self.w * 0.40), int(self.h * 0.65)), anchor=W, font=('Arial', 12), text="Pathology: {}".format(pathology))
        
        def selector_buttons():
            self.selector_sa_btn.place(x=int(self.w * 0.1), y=(self.h * 0.70), width=self.w // 20, height=self.h // 20)
            self.selector_sale_btn.place(x=int(self.w * 0.16), y=(self.h * 0.70), width=self.w // 20, height=self.h // 20)
            self.selector_la_btn.place(x=int(self.w * 0.22), y=(self.h * 0.70), width=self.w // 20, height=self.h // 20)
            self.selector_lale_btn.place(x=int(self.w * 0.28), y=(self.h * 0.70), width=self.w // 20, height=self.h // 20)
        
        def scrolled_text():
            self.anamnezis.place(x=int(self.w * 0.05), y=(self.h * 0.82), width=int(self.w * 0.25), height=int(self.h * 0.1))
            self.anamnezis.configure(state="normal")
            self.anamnezis.delete(1.0, END)
            self.anamnezis.insert(INSERT, self.page.text)
            self.anamnezis.configure(state="disabled")

        login_widgets()
        widget_sizes()
        draw_squares()
        draw_splitting_line()
        show_image()
        name_page_label()
        scoring_buttons()
        selector_buttons()
        scrolled_text()
        self._resize_button()
    
    def last_page(self):
        self.scoring1_btn.destroy()
        self.scoring2_btn.destroy()
        self.scoring3_btn.destroy()
        self.scoring4_btn.destroy()
        self.scoring5_btn.destroy()
        self.scoring6_btn.destroy()
        self.scoring7_btn.destroy()
        self.scoring8_btn.destroy()
        self.scoring9_btn.destroy()
        self.scoring10_btn.destroy()
        self.selector_sa_btn.destroy()
        self.selector_sale_btn.destroy()
        self.selector_la_btn.destroy()
        self.selector_lale_btn.destroy()
        self.anamnezis.destroy()
        self.next_button.destroy()
        if self.login_entry is not None:
            self.login_entry.destroy()
        if self.login_label is not None:
            self.login_label.destroy()
        self.canvas.create_text((int(self.w * 0.5), int(self.h * 0.5)), font=("Times", 40), text="Thank you for your help!")
    
    # WIDGET HANDLERS
    def _resize_button(self):
        self.next_button.place(x=int(self.w * 0.85), y=(self.h * 0.85), width=self.w // 10, height=self.h // 10)


class PersistenceManger:
    
    def __init__(self, page, entry):
        self.page = page
        self.entry = entry
        self.user_file = None
    
    def load_info(self):
        scoring_folder = os.path.join(DATA_PATH, 'scores')
        self.user_file = open(os.path.join(scoring_folder, self.entry.get()) + '.txt', 'a+')
        temp = open(os.path.join(scoring_folder, self.entry.get()) + '.txt', 'rt')
        self.page.pages = len(os.listdir(DATA_PATH)) - 1
        self.page.page_num = len(temp.readlines()) - 1
        temp.close()
    
    def save_score(self, score):
        self.user_file.write("{} {} ({})\n".format(self.page.page_num, score, self.page.get_perm_page_num()))
        self.user_file.flush()

    def next_is_last(self):
        return self.page.page_num >= self.page.pages
    
    def load_images(self):
        if self.page.page_num >= self.page.pages:
            return 
        self.page.paths = dict()
        self.page.num_images = dict()
        path = os.path.join(DATA_PATH, str(self.page.get_perm_page_num()))
        file_names = [fn for fn in os.listdir(path) if fn != "anam.txt"]
        for name in file_names:
            view = name.split('_')[0]
            seqidx = int(name.split('_')[1].split('.')[0])
            if view not in self.page.paths:
                self.page.paths[view] = dict()
            self.page.paths[view][seqidx] = os.path.join(path, name)
        for view in self.page.paths:
            self.page.num_images[view] = len(self.page.paths[view])
    
    def load_text(self):
        if self.page.page_num >= self.page.pages:
            return 
        txt_path = os.path.join(DATA_PATH, str(self.page.get_perm_page_num()), "anam.txt")
        self.page.text = ""
        if not os.path.exists(txt_path):
            self.page.text = "nem áll rendelkezésre adat"
            return
        with open(txt_path, 'rt', encoding='utf8') as txt:
            for line in txt:
                self.page.text += line

    def transition(self, state, action):
        if action == Action.PAGING and state == State.LOGIN:
            self.load_info()
            self.page.page_num += 1
            self.load_images()
            self.load_text()
        elif action == Action.PAGING and state == State.SCORED:
            self.save_score(self.page.scoring)
            self.page.page_num += 1
            self.load_images()
            self.load_text()
        
    def __del__(self):
        if self.user_file != None:
            self.user_file.close()


class CanvasUpdater:
    
    def __init__(self, master):
        self.master = master
        self.info = None
        self.paths = None
        
        default_width = 400
        default_height = 300
        self.canvas = Canvas(master, background="#a2bce5", width=default_width, height=default_height)
        self.canvas.pack(fill=BOTH, expand=1)
        
        widgets = []
        self.next_button = Button(self.master, text="Next", anchor=CENTER, command=self.next_button_pressed)
        widgets.append(self.next_button)
        self.login_entry = Entry(self.master)
        widgets.append(self.login_entry)
        self.login_label = Label(self.master, text="Login as:")
        widgets.append(self.login_label)
        self.pathology1_btn = Button(self.master, text=pathologies[1], anchor=CENTER, command=self.pathology_button_gen(1))
        widgets.append(self.pathology1_btn)
        self.pathology2_btn = Button(self.master, text=pathologies[2], anchor=CENTER, command=self.pathology_button_gen(2))
        widgets.append(self.pathology2_btn)
        self.pathology3_btn = Button(self.master, text=pathologies[3], anchor=CENTER, command=self.pathology_button_gen(3))
        widgets.append(self.pathology3_btn)
        self.pathology4_btn = Button(self.master, text=pathologies[4], anchor=CENTER, command=self.pathology_button_gen(4))
        widgets.append(self.pathology4_btn)
        self.pathology5_btn = Button(self.master, text=pathologies[5], anchor=CENTER, command=self.pathology_button_gen(5))
        widgets.append(self.pathology5_btn)
        self.pathology6_btn = Button(self.master, text=pathologies[6], anchor=CENTER, command=self.pathology_button_gen(6))
        widgets.append(self.pathology6_btn)
        self.pathology7_btn = Button(self.master, text=pathologies[7], anchor=CENTER, command=self.pathology_button_gen(7))
        widgets.append(self.pathology7_btn)
        self.pathology8_btn = Button(self.master, text=pathologies[8], anchor=CENTER, command=self.pathology_button_gen(8))
        widgets.append(self.pathology8_btn)
        self.pathology9_btn = Button(self.master, text=pathologies[9], anchor=CENTER, command=self.pathology_button_gen(9))
        widgets.append(self.pathology9_btn)
        self.pathology10_btn = Button(self.master, text=pathologies[10], anchor=CENTER, command=self.pathology_button_gen(10))
        widgets.append(self.pathology10_btn)
        self.selector_sa_btn = Button(self.master, text="SA", anchor=CENTER, command=self.select_button_gen(View.SA.value))
        widgets.append(self.selector_sa_btn)
        self.selector_sale_btn = Button(self.master, text="SALE", anchor=CENTER, command=self.select_button_gen(View.SALE.value))
        widgets.append(self.selector_sale_btn)
        self.selector_la_btn = Button(self.master, text="LA", anchor=CENTER, command=self.select_button_gen(View.LA.value))
        widgets.append(self.selector_la_btn)
        self.selector_lale_btn = Button(self.master, text="LALE", anchor=CENTER, command=self.select_button_gen(View.LALE.value))
        widgets.append(self.selector_lale_btn)
        # creating a text area widget
        scrollbar = Scrollbar(self.master) 
        self.anamnezis = Text(self.master, wrap="word", yscrollcommand=scrollbar.set, borderwidth=0, highlightthickness=0)
        scrollbar.config(command=self.anamnezis.yview)
        widgets.append(self.anamnezis)

        self.canvas.bind("<Configure>", self.configure)

        self.page = Page()
        self.cntmg = ContentManager(self.canvas, widgets, self.page)
        self.ptncmg = PersistenceManger(self.page, self.login_entry)
        self.state = StateManager(self.cntmg, self.ptncmg)

        self.state.transition(Action.NEUTRAL, default_width, default_height)
    
    def key_left(self, event):
        current_vertical_idx = self.page.chosen_image[0]
        current_horizontal_idx = self.page.chosen_image[1]
        if current_horizontal_idx > 0:
            current_horizontal_idx -= 1
        elif current_vertical_idx > 0:
            current_horizontal_idx = 24
            current_vertical_idx -= 1
        elif self.cntmg.view in self.page.num_images:
            num_images = self.page.num_images[self.cntmg.view]
            current_horizontal_idx = (num_images - 1) % 25
            current_vertical_idx = (num_images - 1) // 25
        self.page.chosen_image = (current_vertical_idx, current_horizontal_idx)
        self.state.transition(Action.NEUTRAL, self.cntmg.w, self.cntmg.h)
    def key_down(self, event):
        current_vertical_idx = self.page.chosen_image[0]
        if self.cntmg.view in self.page.num_images and ((current_vertical_idx + 1) * 25 + self.page.chosen_image[1]) < self.page.num_images[self.cntmg.view]:
            current_vertical_idx += 1
        self.page.chosen_image = (current_vertical_idx, self.page.chosen_image[1])
        self.state.transition(Action.NEUTRAL, self.cntmg.w, self.cntmg.h)
    def key_right(self, event):
        current_vertical_idx = self.page.chosen_image[0]
        current_horizontal_idx = self.page.chosen_image[1]
        if self.cntmg.view in self.page.num_images and (self.page.chosen_image[0] * 25 + current_horizontal_idx + 1) < self.page.num_images[self.cntmg.view] and (current_horizontal_idx < 24):
            current_horizontal_idx += 1
        elif self.cntmg.view in self.page.num_images and ((self.page.chosen_image[0] + 1) * 25 < self.page.num_images[self.cntmg.view]):
            current_horizontal_idx = 0
            current_vertical_idx += 1
        else:
            current_horizontal_idx = 0
            current_vertical_idx = 0
        self.page.chosen_image = (current_vertical_idx, current_horizontal_idx)
        self.state.transition(Action.NEUTRAL, self.cntmg.w, self.cntmg.h)
    def key_up(self, event):
        self.page.chosen_image = (max(self.page.chosen_image[0] - 1, 0), self.page.chosen_image[1])
        self.state.transition(Action.NEUTRAL, self.cntmg.w, self.cntmg.h)
        
    def next_button_pressed(self):
        self.state.transition(Action.PAGING, self.cntmg.w, self.cntmg.h)
    
    def pathology_button_gen(self, score):
        def command_func():
            self.state.transition(Action(score + 1), self.cntmg.w, self.cntmg.h)
        return command_func
    
    def select_button_gen(self, view):
        def command_func():
            self.cntmg.view = view
            self.page.chosen_image = (0, 0)
            self.state.transition(Action.NEUTRAL, self.cntmg.w, self.cntmg.h)
        return command_func

    def configure(self, event):
        self.state.transition(Action.NEUTRAL, event.width, event.height)


class Window:

    # Define settings upon initialization. Here you can specify
    def __init__(self, root=None):

        # reference to the master widget, which is the tk window
        self.root = root
        self.frame = Frame(root)
        self.folder_name = None

        # with that, we want to then run init_window, which doesn't yet exist
        self.init_window()
        self.adding_canvas()

    # Creation of init_window
    def init_window(self):

        # changing the title of our master widget      
        self.root.title("GUI")

        # allowing the widget to take the full space of the root window
        self.frame.pack(fill=BOTH, expand=1)
    
    def adding_canvas(self):
        # Instantiate the canvas object here
        self.canvas = CanvasUpdater(self.frame)
        self.root.bind("<Up>", self.canvas.key_up)
        self.root.bind("<Down>", self.canvas.key_down)
        self.root.bind("<Left>", self.canvas.key_left)
        self.root.bind("<Right>", self.canvas.key_right)


def start_gui():
    # root window created. Here, that would be the only window, but
    # you can later have windows within windows. 
    root = Tk()
    root.geometry("400x300")
    #creation of an instance
    app = Window(root)
    #mainloop 
    root.mainloop()

start_gui()
