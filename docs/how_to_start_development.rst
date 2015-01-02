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
`liboptv/include/parameters.h https://github.com/OpenPTV/openptv/blob/master/liboptv/include/parameters.h`_ and find the mm_np struct.

2. complete a compare_mm_np() function following the example of the other
parameters structures and use it in compare_control_par().

3. following the example in pybind/, write a Python wrapper for mm_np, and add
a test for it in tests/

4. We work with the git source-code management system. You can do a pull
request for your work against my repository,
https://github.com/yosefm/openptv
or you can email me back a patch series produced by git if you are reluctant
to open a Github account before getting the job.

5. Bonus points for allowing a Python user to read the parameters file from
Python.

Naturally, not a lot of deep-C code will be needed for this one task so some
of you may see it as below their skills - don't worry, it gets more
complicated :)

If you require some additional guidance on how to get the job done, or what's
exactly happening in the code, please contact me directly.

Good luck!
