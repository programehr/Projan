import argparse

class DictReader(argparse.Action):
    def __init__(self,
                 option_strings,
                 dest,
                 nargs=None,
                 const=None,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None,
                 type_map=None):
        self.type_map = type_map

        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        import re
        # arr = values[0].split(' ')
        arr = re.split(r'(?:\s*,{,1}\s+)|(?:\s+,{,1}\s*)|,', values[0])
        dic = {}
        for x in arr:
            k, v = x.split('=')
            if k in self.type_map:
                vtype = self.type_map[k]
                v = vtype(v)
            dic[k] = v
        if hasattr(namespace, self.dest):
            old_vals = getattr(namespace, self.dest)
        else:
            old_vals = []
        old_vals = [] if old_vals is None else old_vals
        new_vals = old_vals + [dic]
        setattr(namespace, self.dest, new_vals)
