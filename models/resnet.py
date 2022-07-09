#!/usr/bin/python3

import sys
sys.path.append('../')
from pycore.module import (
    to_BatchNorm,
    to_begin,
    to_Conv,
    to_cor,
    to_end,
    to_generate,
    to_head,
    to_input,
    to_Relu,
)


arch = [ 
    to_head('..'), 
    to_cor(),
    to_begin(),
    
    #input
    to_input( '../examples/fcn8s/cats.jpg' ),

    to_Conv(name='conv0', s_filer=224, n_filer=3, offset="(0,0,0)", to="(0,0,0)", width=3, height=40, depth=40, caption=""),
    to_BatchNorm(name='bn0', offset="(0,0,0)", to="(conv0-east)", width=3, height=40, depth=40, opacity=0.5, caption=""),
    to_Relu(name='relu0', offset="(0,0,0)", to="(bn0-east)", width=3, height=40, depth=40, opacity=0.5, caption=""),
    to_end() 
    ]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
    
