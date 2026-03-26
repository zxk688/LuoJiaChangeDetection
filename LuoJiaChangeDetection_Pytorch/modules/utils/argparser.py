import argparse


def get_argparser():
    parser = argparse.ArgumentParser(
                    prog='Change_Detection_Framework',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('--config',type=str,default='./configs/resunet.yml',required=False,help="Directory of the config file")
    return parser

