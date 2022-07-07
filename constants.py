import os
import sys


IS_WINDOWS = sys.platform == 'win32'
get_trace = getattr(sys, 'gettrace', None)
DEBUG = get_trace is not None and get_trace() is not None
EPSILON = 1e-4
DIM = 3
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
DATA_ROOT = f'{PROJECT_ROOT}/assets/'
RAW_ROOT = f'{DATA_ROOT}raw/'
PLOT_ROOT = f'{PROJECT_ROOT}/plots/'
OUT_ROOT = f'{DATA_ROOT}out/'
RENDERS_ROOT = f'{DATA_ROOT}renders/'
CHECKPOINTS_ROOT = f'{DATA_ROOT}checkpoints/'
FLASK_PUBLIC = 'C:/Users/dori2/Desktop/Bezalel/Year 5/pgmr/website/react-3d/public/'
ARCHIVE_IMAGES = 'C:/Users/dori2/Desktop/Bezalel/Year 5/pgmr/website/react-3d/archive_images/'
ARCHIVE_MODELS = 'C:/Users/dori2/Desktop/Bezalel/Year 5/pgmr/website/react-3d/archive_models/'
ARCHIVE_JSON = 'C:/Users/dori2/Desktop/Bezalel/Year 5/pgmr/website/react-3d/src/resources/'
CACHE_ROOT = f'{DATA_ROOT}cache/'
UI_OUT = f'{DATA_ROOT}ui_export/'
POINTS_CACHE = f'{DATA_ROOT}points_cache/'
DualSdfData =  f'{RAW_ROOT}dualSDF/'
UI_RESOURCES = f'{DATA_ROOT}/ui_resources/'

mnt = '/mnt/amir' if os.path.isdir('/mnt/amir') else '/data/amir'
Shapenet_WT = f'{mnt}/ShapeNetCore_wt/'
Shapenet = f'{mnt}/ShapeNetCore.v2/'
PARTNET_ROOT = f'{mnt}/partnet_seg/'

COLORS = [[231, 231, 91], [103, 157, 200], [177, 116, 76], [88, 164, 149],
         [236, 150, 130], [80, 176, 70], [108, 136, 66], [78, 78, 75],
         [41, 44, 104], [217, 49, 50], [87, 40, 96], [85, 109, 115], [234, 234, 230],
          [30, 30, 30]]
GLOBAL_SCALE = 10
IM_NET_DS = f'{RAW_ROOT}IM-NET/'
MAX_GAUSIANS = 32
MAX_VS = 100000

