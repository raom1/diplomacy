from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox



def render_map():
    x = []
    y = []
    for l, t in zip(units_df['location'], units_df['type']):
        c = territories_df.loc[territories_df['country'] == l, 'coords'].values[0][t]
        x.append(c[0])
        y.append(c[1])
    
    x, y = np.atleast_1d(x, y)
    
    fig, ax = plt.subplots(figsize = (22.7, 19.51))

    dip_map = plt.imread('map_assets/diplomacy_map.gif')
    dip_im = OffsetImage(dip_map, zoom = 1)
    dip_ab = AnnotationBbox(dip_im, (0, 0), xycoords = 'data', frameon = False, box_alignment = (0, 0))
    ax.add_artist(dip_ab)
#     fleet_icon = plt.imread('personal/fleet_icon.png')
    fleet_icon_dict = {c: OffsetImage(plt.imread('map_assets/fleet_{}.png'.format(c)), zoom = 1) for c in units_df['owner'].unique()}
    army_icon_dict = {c: OffsetImage(plt.imread('map_assets/army_{}.png'.format(c)), zoom = 1) for c in units_df['owner'].unique()}
#     fleet_im = OffsetImage(fleet_icon, zoom = 1)
#     army_icon = plt.imread('personal/army_icon.png')
#     army_im = OffsetImage(army_icon, zoom = 1)
    for _, row in units_df.iterrows():
        x0, y0 = territories[row['location']]['coords'][row['type']]
        if row['type'] == 'army':
            ab = AnnotationBbox(army_icon_dict[row['owner']], (x0, 965-y0), xycoords = 'data', frameon = False)
        else:
            ab = AnnotationBbox(fleet_icon_dict[row['owner']], (x0, 965-y0), xycoords = 'data', frameon = False)
        ax.add_artist(ab)
    ax.update_datalim(np.column_stack([[0, 1152], [0, 965]]))
    ax.autoscale()
    plt.axis('off')
    plt.show();