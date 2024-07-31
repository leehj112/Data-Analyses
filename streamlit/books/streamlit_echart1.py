# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:29:45 2024

@author: leehj
"""

# https://echarts.streamlit.app/
# pip install streamlit_echarts

import streamlit as st
import streamlit_echarts as echarts

options = {
    "xAxis": {
        "type": "category",
        "boundaryGap": False,
        "data": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    },
    "yAxis": {"type": "value"},
    "series": [
        {
            "data": [820, 932, 901, 934, 1290, 1330, 1320],
            "type": "line",
            "areaStyle": {},
        }
    ],
}
echarts.st_echarts(options=options)
