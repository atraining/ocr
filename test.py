#!/usr/bin/env python
# -*- coding: utf-8 -*-
# from DDR.test import test_gen_sldwins, test_gen_task_gts, test_sldwins, test_task_gts
from RRCNN.test import test_gen_task_gts
# from ICDAR.test import gen_data
import os
from ONLINE.utils import parse_online, gen_line_quad
from ONLINE.annos_check import analysis_annos
from ONLINE.line_utils import gen_line
from icdar2017rctw.line_utils import gen_line_rctw
from icdar2017rctw.utilities import gen_data

if __name__ == '__main__':
    # gen_sldwins()
    # test_sldwins()
    # test_gen_task_gts()
    # test_task_gts()
    # gen_data()
    # test_gen_task_gts()
    # parse_online()
    # gen_line()
    gen_line_quad()
    # gen_line_rctw()
    # gen_data()
    # analysis_annos()