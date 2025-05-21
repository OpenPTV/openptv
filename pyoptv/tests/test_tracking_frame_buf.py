import pytest
import numpy as np
from pyoptv import tracking_frame_buf

def test_initialize_frame_buffer():
    frame_buffer = tracking_frame_buf.initialize_frame_buffer(10)
    assert len(frame_buffer) == 10
    assert all(isinstance(frame, list) for frame in frame_buffer)

def test_add_to_frame_buffer():
    frame_buffer = tracking_frame_buf.initialize_frame_buffer(10)
    point = (1.0, 2.0, 3.0)
    tracking_frame_buf.add_to_frame_buffer(frame_buffer, point, 0)
    assert frame_buffer[0] == [point]

def test_get_points_from_frame_buffer():
    frame_buffer = tracking_frame_buf.initialize_frame_buffer(10)
    points = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
    tracking_frame_buf.add_to_frame_buffer(frame_buffer, points[0], 0)
    tracking_frame_buf.add_to_frame_buffer(frame_buffer, points[1], 0)
    retrieved_points = tracking_frame_buf.get_points_from_frame_buffer(frame_buffer, 0)
    assert retrieved_points == points

def test_clear_frame_buffer():
    frame_buffer = tracking_frame_buf.initialize_frame_buffer(10)
    point = (1.0, 2.0, 3.0)
    tracking_frame_buf.add_to_frame_buffer(frame_buffer, point, 0)
    tracking_frame_buf.clear_frame_buffer(frame_buffer, 0)
    assert frame_buffer[0] == []

def test_frame_buffer_full():
    frame_buffer = tracking_frame_buf.initialize_frame_buffer(2)
    point = (1.0, 2.0, 3.0)
    tracking_frame_buf.add_to_frame_buffer(frame_buffer, point, 0)
    tracking_frame_buf.add_to_frame_buffer(frame_buffer, point, 1)
    assert tracking_frame_buf.frame_buffer_full(frame_buffer) == True
    tracking_frame_buf.clear_frame_buffer(frame_buffer, 0)
    assert tracking_frame_buf.frame_buffer_full(frame_buffer) == False
