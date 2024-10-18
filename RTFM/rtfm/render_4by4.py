import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.transform import resize
#import rtfm
import copy

def cwd():
    print(os.path.dirname(__file__))
    
def load_sprites(main_dir, show=True):
    sprite_dict = {}
    dirs = os.listdir(main_dir)
    for d in dirs:
        if d[0] != '.':
            if show:
                print(d[:-4])
            x = np.load(main_dir+d)
            if show:
                print(x.shape)
                plt.imshow(x)
                plt.show()
            sprite_dict[d[:-4]] = x
    return sprite_dict

def render_frame_from_rb(frames, gym_env, b, t):
    f = {}
    for k in frames.keys():
        f[k] = frames[k][b,t]
    render_frame(f, gym_env)
    
def render_frame(frame, gym_env, **kwargs):
    path = os.path.dirname(__file__)
    items_dict = load_sprites(path+'/Sprites/Items/', show=False)
    monster_dict = load_sprites(path+'/Sprites/Monsters/', show=False)
    agent_dict = load_sprites(path+'/Sprites/Agent/', show=False)
    background_dict = load_sprites(path+'/Sprites/Background/', show=False)
    sprites_dict = {**items_dict, **monster_dict, **agent_dict, **background_dict}
    
    frame_dict, object_pos_dict = process_env_frame(frame, gym_env)
    canvas = get_canvas(frame_dict, sprites_dict)
    plot_canvas(canvas, object_pos_dict, sprites_dict, frame, gym_env, **kwargs)
    
def process_env_frame(frame, gym_env):
    """
    This has the limitation of looking only at the first placement.
    """
    x = frame['name']
    assert len(x.shape) == 4, 'frame[name] has the wrong number of dimensions'
    H, W = x.shape[:2]
    if H !=4:
        x_HW = x[1:-1,1:-1]
    else:
        x_HW = x
    vocab = gym_env.vocab
    frame_nl = np.array([['_'*12,'_'*12] for i in range(16)]).reshape(4,4,2)
    object_pos_dict = {}
    for i in range(4):
        for j in range(4):
            # first and second token of the first object
            w1 = vocab.index2word(x_HW[i,j,0,0])
            w2 = vocab.index2word(x_HW[i,j,0,1])
            # first and second token of the second object (if overlapping)
            w3 = vocab.index2word(x_HW[i,j,1,0])
            w4 = vocab.index2word(x_HW[i,j,1,1])
            if w1 != 'empty':
                if w1 == 'you' and w3 != 'empty':
                    key = 'you, \n%s %s'%(w3,w4)
                elif w1 == 'you' and w3 == 'empty':
                    key = 'you'
                elif w1 != 'you' and w3 != 'empty':
                    key = '%s %s, \n%s %s'%(w1,w2,w3,w4)
                elif w1 != 'you' and w3 == 'empty':
                    key = '%s %s'%(w1,w2)
                else:
                    raise NotImplementedError
                object_pos_dict[key] = [i,j]
                
            frame_nl[i,j] = [w1,w2]
            
    frame_dict = {'name':frame_nl, 'inv':frame['name']}
    return frame_dict, object_pos_dict
    
def get_canvas(frame_dict, sprites_dict):
    frame_HW = frame_dict['name']
    inv = frame_dict['inv']
    H = frame_HW.shape[0]
    W = frame_HW.shape[1]
    res = 64
    C = 4
    canvas_small = copy.deepcopy(sprites_dict['background']) #64x64
    canvas = np.tile(canvas_small, (4,4,1)) # 256x256
    for i in range(frame_HW.shape[0]):
        for j in range(frame_HW.shape[1]):
            if frame_HW[i,j,0] == 'you':
                k = 'you'
            else:
                k = frame_HW[i,j,1]
            if k in sprites_dict.keys():
                x = sprites_dict[k]
                mask = (x[:,:,3]==0.).reshape(res,res,1).astype(float)
                canvas[i*res:(i+1)*res, j*res:(j+1)*res] = canvas[i*res:(i+1)*res, j*res:(j+1)*res]*mask + (1-mask)*x
    return canvas

def get_inventory_canvas(frame, sprites_dict, gym_env):
    res = 64
    canvas = sprites_dict['inventory']
    k = gym_env.vocab.index2word(frame['inv'][1])
    if k in sprites_dict.keys():
        x = sprites_dict[k]
        mask = (x[:,:,3]==0.).reshape(res,res,1).astype(float)
        canvas = canvas*mask + (1-mask)*x
    return canvas
    
def plot_canvas(canvas, object_pos_dict, sprites_dict, frame, gym_env, fig_H=8, fig_W=8):
    plt.figure(figsize=(fig_H, fig_W))
    ax = plt.subplot2grid((5,5), (0, 0), colspan=4, rowspan=4)
    fig_H *= 4/5 
    fig_W *= 4/5
    ax.imshow(canvas)
    ax.set_xticks(np.array([63,127,191]), minor=True)
    ax.set_yticks(np.array([63,127,191]), minor=True)
    ax.grid(which="minor", color='black')
    ax.tick_params(which="minor", size=0)
    plt.xticks([])
    plt.yticks([])
    for k in object_pos_dict.keys():
        x_coord, y_coord = get_axes_fraction(object_pos_dict[k], k)
        plt.annotate('[%s]'%k, (x_coord, y_coord),  xycoords = 'axes fraction', color='black', backgroundcolor='white', fontsize=int(fig_H*1.4))

    wiki = 'Wiki: \n'+translate_in_nl(frame['wiki'], gym_env.vocab)
    goal = 'Goal: '+translate_in_nl(frame['task'], gym_env.vocab)
    plt.title(goal+'\n'+wiki, fontsize=int(fig_H*1.4))
    
    
    ax_inv = plt.subplot2grid((5,5), (0, 4), colspan=1, rowspan=1)
    inv_canvas = get_inventory_canvas(frame, sprites_dict, gym_env)
    ax_inv.imshow(inv_canvas) # change this to the correct item
    plt.xticks([])
    plt.yticks([])
    inv_content = translate_in_nl(frame['inv'], gym_env.vocab)
    if inv_content == '':
        inv_content = 'empty'
    plt.title('Inventory: \n'+inv_content, fontsize=int(fig_H*1.2))

    plt.tight_layout()
    plt.show()
    
def get_axes_fraction(pos, word):
    y_coord = 0.75 - 0.25*pos[0] + 0.02
    x_coord = 0.25*pos[1] +0.25/18*(max(16-len(word),1))/2
    return x_coord, y_coord

def translate_in_nl(sequence, vocab):
    seq_nl = ''
    for w in sequence:
        if vocab.index2word(w) != 'pad':
            seq_nl += ' '+vocab.index2word(w)
        if len(seq_nl) > 0:
            if seq_nl[-1] == '.':
                seq_nl += '\n'
    return seq_nl


