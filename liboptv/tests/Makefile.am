TESTS = check_fb
check_PROGRAMS = check_fb
dist_noinst_DATA = testing_fodder/*

check_fb_SOURCES = check_fb.c $(top_builddir)/include/tracking_frame_buf.h
check_fb_CFLAGS = @CHECK_CFLAGS@ -I../include/
check_fb_LDADD = $(top_builddir)/src/liboptv.la @CHECK_LIBS@

