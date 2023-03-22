import yaml
import argparse
import random
from utils.model import ImageTranslationBranch
from utils.base_util import *
import time

H, W = 0, 0
def fetch_IT_input(folder_pos):
    multi_smooth_list = ['smooth_0', 'smooth_1', 'smooth_2', 'smooth_3', 'smooth_4', 'smooth_5']

    depth_path = os.path.join(folder_pos, 'depth.png')
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / 65535.0
    H, W = depth_img.shape
    cv2.resize(depth_img, dsize=(768, 768), interpolation=cv2.INTER_CUBIC)
    dim1 = depth_img.shape[0]
    dim2 = depth_img.shape[1]
    depth_img = depth_img[np.newaxis, :, :] - 0.906764  # subtract average

    nv_img = np.zeros((len(multi_smooth_list), dim1, dim2))
    for smooth_id, smooth_str in enumerate(multi_smooth_list):
        nv_path = os.path.join(folder_pos, '%s.png' % smooth_str)
        smooth_i = cv2.imread(nv_path, cv2.IMREAD_UNCHANGED) / 65535.0 - 0.974258
        cv2.resize(smooth_i, dsize=(768, 768), interpolation=cv2.INTER_CUBIC)
        nv_img[smooth_id, :, :] = smooth_i

    output_np = np.concatenate((nv_img, depth_img), axis=0)
    output_np = output_np[np.newaxis, :, :, :]
    output = torch.from_numpy(output_np).type(torch.cuda.FloatTensor)
    return output

def show_img(cv2_array, ofn=None):
    if ofn is None:
        cv2.imshow('image', cv2_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.resize(smooth_i, dsize=(W, H), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(ofn, cv2_array)

parser = argparse.ArgumentParser()
parser.add_argument('-model_name', type=str, required=True)
parser.add_argument('-config_file', type=str, default='configs/default_config.yml')
parser.add_argument('-save_name', type=str, required=True)
args = parser.parse_args()

conf = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)
start_time = time.time()
torch.manual_seed(conf['seed_num'])
torch.cuda.manual_seed(conf['seed_num'])
torch.cuda.manual_seed_all(conf['seed_num'])  # if you are using multi-GPU.
np.random.seed(conf['seed_num'])  # Numpy module.
random.seed(conf['seed_num'])  # Python random module.
torch.manual_seed(conf['seed_num'])
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

folder_pos = os.path.join(conf['data_folder'], args.model_name)
with torch.no_grad():
    IT_branch = ImageTranslationBranch(7, 1, ngf=64, n_downsampling=4, n_blocks=9)
    IT_branch.cuda()
    IT_branch.eval()
    IT_branch.load_state_dict(torch.load(conf['ITB_path']))
    IT_input = fetch_IT_input(folder_pos)
    IT_output_probability = IT_branch(IT_input)
    IT_output_np = np.squeeze(IT_output_probability.data.cpu().float().numpy())
    # IT_output_with_base = ridge_detection(IT_output_np, folder_pos, conf)

show_tensor(IT_output_np, args.save_name, crop=True)