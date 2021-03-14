import argparse
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

# local:
from include.Config import Config
from include.Model import build, training
from include.Test import get_hits
from include.Load import *

'''
Follow the code style of GCN-Align:
https://github.com/1049451037/GCN-Align
'''

seed = 12306
np.random.seed(seed)
tf.random.set_seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argparser')
    parser.add_argument('--language',
                        default='ja_en',
                        help='Languages of the knowledge graphs. Viable options are limited to:\n'
                             + "'ja_en', 'zh_en', 'ja_en', 'fr_en', 'dpb_yg'.")
    parser.add_argument('--epochs',
                        default=600,
                        help='Number of training epochs of the GCN part of the architecture.')
    args = parser.parse_args()
    config_obj = Config(language=args.language, epochs=int(args.epochs))

    e = len(set(loadfile(config_obj.e1, 1)) | set(loadfile(config_obj.e2, 1)))

    ILL = loadfile(config_obj.ill, 2)
    illL = len(ILL)
    np.random.shuffle(ILL)
    train = np.array(ILL[:illL // 10 * Config.seed])
    test = ILL[illL // 10 * Config.seed:]

    KG1 = loadfile(config_obj.kg1, 3)
    KG2 = loadfile(config_obj.kg2, 3)

    output_layer, loss = build(
        Config.dim, Config.act_func, Config.alpha, Config.beta, Config.gamma, Config.k,
        config_obj.language.split('_', 1)[0], e, train, KG1 + KG2)

    vec, J = training(output_layer, loss, 0.001, config_obj.epochs, train, e, Config.k, test,
                      language=config_obj.language)
    print('loss:', J)
    print('Result:')
    get_hits(vec, test)
