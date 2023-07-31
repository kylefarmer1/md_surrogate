"""
Input deck reader which combines the utility of argparse and yaml. yaml provides the ability to read a simple
key-value input deck while argparse provides the ability to modify variables at the command line. This script can be
utilized as the following:

The Arguments object should be treated like the argparse.ArgumentParser object, i.e. in your code add the following to
get access to defined variables:

from args import Arguments
arg_parser = Arguments()
args = arg_parser.parse_args()

The available arguments are all contained within the default yaml config file: 'config.yaml'. If a key is not contained
in the default yaml file, it cannot be created elsewhere. Nested config files will be recursively converted to
Namespaces, i.e. args.a.b.c = 2, args.a.b.d = 3.
The values of the arguments can be modified in 4 different ways:
1. modify 'config.yaml'
2. pass the key-value pair(s) at the command line, i.e. python script.py --num_time_steps 100 --cutoff 5
3. create a second .yaml file and pass it to the --config command line option,
    i.e. python script.py --config config2.yaml
4. pass a second .yaml file to the --config option and subsequently add other command line options (must be done in
    this order, else --config will override it). i.e. python script.py --config config2.yaml --num_time_steps 100


"""
import argparse
import yaml
DEFAULT_CONFIG = './config.yaml'
STRING_OPTIONS = [
    'model_save_directory'
]


class Arguments:
    """argparse.ArgumentsParser-like object to include yaml config options"""
    def __init__(self):
        """Initialize Arguments parser

        The default config file (`DEFAULT_CONFIG`) is opened and each key is added as an argument which can be accessed at the
        command line, along with a 'config' option to read in a second '.yaml' config file..
        """
        self.default_args = yaml.safe_load(open(DEFAULT_CONFIG, 'r'))
        parser = argparse.ArgumentParser()
        parser.add_argument("--config")
        for k in self.default_args.keys():
            type = float if k not in STRING_OPTIONS else str
            parser.add_argument("--"+str(k), type=type)
        self.parser = parser

    def parse_args(self):
        """Parse the arguments contained in yaml and command line options

        Add the yaml options to the args object. Update the args object with the second 'yaml' config file (if passed),
        then add the command line options.

        :return: argparse.Namespace object -> dict-like object where key value pairs can be accessed with:
                `args.key = value`
        """
        args = self.default_args
        cli_args = self.parser.parse_args()
        if cli_args.config:
            config = yaml.safe_load(open(cli_args.config, 'r'))
            args.update(config)

        # remove any None entries
        cli_args = vars(cli_args)
        cli_args_filtered = {k: v for k, v in cli_args.items() if v is not None}
        args.update(cli_args_filtered)
        args = Arguments.dict_to_namespace(args)

        return args

    @staticmethod
    def dict_to_namespace(d):
        """Recursively set Namespace from nested dicts

        :param d: Dictionary (nested is acceptable)
        :return: argparse.Namespace
        """
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = argparse.Namespace(**v)
                d[k] = Arguments.dict_to_namespace(v)
        return argparse.Namespace(**d)

    @staticmethod
    def print_args(args):
        """Helper function for printing the compiled args"""
        print(f"Configurations\n{'=' * 50}")
        [print(k, ':', v) for k, v in vars(args).items()]
        print('=' * 50)
