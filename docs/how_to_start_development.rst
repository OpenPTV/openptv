Dear new OpenPTV developer,

Thank you for starting with OpenPTV development. A lot of research groups, both in academy and industry, 
would be grateful to use an open source software for 3D particle tracking. We will work with us on transforming the
existing code to something we all can be proud of.

To allow us to get on the same page, we need to proceed with this small introductory test that will also
give you an impression of the nature of work we need done. The task is outlined below: 

The steps are simple: <https://github.com/OpenPTV/openptv> is where our code is. We work with the ``git`` source-code management system. You can do a pull
request for your work against my repository, `https://github.com/yosefm/openptv <https://github.com/yosefm/openptv>`_
or you can email @yosefm a patch series produced by git if you are reluctant to open a Github account before getting the job.

1.  Look at `liboptv/include/parameters.h <https://github.com/OpenPTV/openptv/blob/master/liboptv/include/parameters.h>`_ and 
find the ``mm_np struct``.

2. complete a ``compare_mm_np()`` function in `src/parameters.c <https://github.com/OpenPTV/openptv/blob/master/liboptv/src/parameters.c>`_ following the example of the other
parameters structures and use it in ``compare_control_par()``. Add documentation comments to the ``parameters.c`` following the examples of other functions.

3. create a test function ``check_parameters.c`` in `liboptv/tests/ <https://github.com/OpenPTV/openptv/tree/master/liboptv/tests>`_ and update the ``CMakeLists.txt`` accordingly

4. following the example in `py_bind/ <https://github.com/OpenPTV/openptv/tree/master/py_bind/optv>_`, write a Python wrapper for ``mm_np``, and add
a test for it in ``tests/``. The wrapper includes API that interfaces C with Python, using `Cython <http://docs.cython.org/src/tutorial/clibraries.html>`_ 

5. Write a function ``read_control_parameters.py`` that allows a Python user to read the parameters file from
Python, using the Cython interface to the function read_control_parameters in the ``liboptv`` C library. 

If you require some additional guidance on how to get the job done, or what's
exactly happening in the code, please contact our mailing list <https://groups.google.com/forum/#!forum/openptv>

Good luck!
