Dear applicant,

Thank you for applying to the position we offer in our laboratory. We are
looking for help in advancing an open source academic software used by our
field, and based on your C and Python skills we hope that you will be able to
provide it. The chosen applicant will work with me on transforming the
existing code to something we can be proud of.

After initial filtering, we still face a lot of applicants with the required
skills. Since some you live quite far away we think it better for both sides
to precede the customary interview with a small skills test that will also
give you an impression of the nature of work we need done. The task is
outlined below. Please inform me in the next few days that you intend to take
it up.

The small task we would like all of you to do is the same. The steps are
simple:

1. https://github.com/OpenPTV/openptv is where our code is. Look at
`liboptv/include/parameters.h <https://github.com/OpenPTV/openptv/blob/master/liboptv/include/parameters.h>`_ and 
find the ``mm_np struct``.

2. complete a ``compare_mm_np()`` function in `src/parameters.c <https://github.com/OpenPTV/openptv/blob/master/liboptv/src/parameters.c>`_ following the example of the other
parameters structures and use it in ``compare_control_par()``. Add documentation comments to the ``parameters.c`` following the examples of other functions.

3. following the example in `py_bind/ <https://github.com/OpenPTV/openptv/tree/master/py_bind/optv>_`, write a Python wrapper for ``mm_np``, and add
a test for it in ``tests/``. The wrapper includes API that interfaces C with Python, using `Cython <http://docs.cython.org/src/tutorial/clibraries.html>`_ 

4. We work with the ``git`` source-code management system. You can do a pull
request for your work against my repository,
`https://github.com/yosefm/openptv <https://github.com/yosefm/openptv>`_
or you can email @yosefm a patch series produced by git if you are reluctant
to open a Github account before getting the job.

5. Write a function ``read_control_parameters.py`` that allows a Python user to read the parameters file from
Python, using the Cython interface to the function read_control_parameters in the ``liboptv`` C library. 

If you require some additional guidance on how to get the job done, or what's
exactly happening in the code, please contact our mailing list <https://groups.google.com/forum/#!forum/openptv>

Good luck!
