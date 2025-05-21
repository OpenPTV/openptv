import pytest
from pyoptv.parameters import (
    read_sequence_par, new_sequence_par, free_sequence_par, compare_sequence_par,
    read_track_par, compare_track_par,
    read_volume_par, compare_volume_par,
    read_control_par, free_control_par, compare_control_par,
    read_target_par, compare_target_par, write_target_par
)

def test_read_sequence_par():
    sp = read_sequence_par("test_data/sequence.par", 2)
    assert sp.num_cams == 2
    assert sp.img_base_name == ["cam1", "cam2"]
    assert sp.first == 1
    assert sp.last == 100

def test_new_sequence_par():
    sp = new_sequence_par(2)
    assert sp.num_cams == 2
    assert sp.img_base_name == ["", ""]
    assert sp.first == 0
    assert sp.last == 0

def test_free_sequence_par():
    sp = new_sequence_par(2)
    free_sequence_par(sp)
    assert sp is None

def test_compare_sequence_par():
    sp1 = new_sequence_par(2)
    sp2 = new_sequence_par(2)
    assert compare_sequence_par(sp1, sp2)

def test_read_track_par():
    tp = read_track_par("test_data/track.par")
    assert tp.dvxmin == -0.1
    assert tp.dvxmax == 0.1
    assert tp.dvymin == -0.1
    assert tp.dvymax == 0.1
    assert tp.dvzmin == -0.1
    assert tp.dvzmax == 0.1
    assert tp.dangle == 5.0
    assert tp.dacc == 0.1
    assert tp.add == 1

def test_compare_track_par():
    tp1 = read_track_par("test_data/track.par")
    tp2 = read_track_par("test_data/track.par")
    assert compare_track_par(tp1, tp2)

def test_read_volume_par():
    vp = read_volume_par("test_data/volume.par")
    assert vp.X_lay == [0.0, 1.0]
    assert vp.Zmin_lay == [0.0, 1.0]
    assert vp.Zmax_lay == [0.0, 1.0]
    assert vp.cnx == 0.1
    assert vp.cny == 0.1
    assert vp.cn == 0.1
    assert vp.csumg == 0.1
    assert vp.corrmin == 0.1
    assert vp.eps0 == 0.1

def test_compare_volume_par():
    vp1 = read_volume_par("test_data/volume.par")
    vp2 = read_volume_par("test_data/volume.par")
    assert compare_volume_par(vp1, vp2)

def test_read_control_par():
    cp = read_control_par("test_data/control.par")
    assert cp.num_cams == 2
    assert cp.img_base_name == ["cam1", "cam2"]
    assert cp.cal_img_base_name == ["cal1", "cal2"]
    assert cp.hp_flag == 1
    assert cp.allCam_flag == 1
    assert cp.tiff_flag == 1
    assert cp.imx == 1000
    assert cp.imy == 1000
    assert cp.pix_x == 0.01
    assert cp.pix_y == 0.01
    assert cp.chfield == 1
    assert cp.mm.n1 == 1.0
    assert cp.mm.n2 == [1.0, 1.0, 1.0]
    assert cp.mm.n3 == 1.0
    assert cp.mm.d == [1.0, 1.0, 1.0]
    assert cp.mm.nlay == 1

def test_free_control_par():
    cp = new_control_par(2)
    free_control_par(cp)
    assert cp is None

def test_compare_control_par():
    cp1 = read_control_par("test_data/control.par")
    cp2 = read_control_par("test_data/control.par")
    assert compare_control_par(cp1, cp2)

def test_read_target_par():
    tp = read_target_par("test_data/target.par")
    assert tp.gvthres == [10, 20, 30, 40]
    assert tp.discont == 1
    assert tp.nnmin == 5
    assert tp.nnmax == 10
    assert tp.nxmin == 0
    assert tp.nxmax == 100
    assert tp.nymin == 0
    assert tp.nymax == 100
    assert tp.sumg_min == 50
    assert tp.cr_sz == 5

def test_compare_target_par():
    tp1 = read_target_par("test_data/target.par")
    tp2 = read_target_par("test_data/target.par")
    assert compare_target_par(tp1, tp2)

def test_write_target_par():
    tp = read_target_par("test_data/target.par")
    write_target_par(tp, "test_data/target_out.par")
    tp_out = read_target_par("test_data/target_out.par")
    assert compare_target_par(tp, tp_out)
