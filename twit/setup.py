
#!/usr/bin/env python

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('twit', parent_package, top_path)

    config.add_subpackage('numpy')
    config.add_subpackage('core')
    return config

if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
