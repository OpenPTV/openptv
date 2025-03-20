versions pytest-8.3.4, python-3.11.11.final.0
invocation_dir=/home/user/Documents/repos/openptv/py_bind/test
cwd=/home/user/Documents/repos/openptv/py_bind/test
args=('--debug', 'test_corresp.py')

  pytest_cmdline_main [hook]
      config: <_pytest.config.Config object at 0x7fa936766950>
    pytest_plugin_registered [hook]
        plugin: <Session  exitstatus='<UNSET>' testsfailed=0 testscollected=0>
        plugin_name: session
        manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
    finish pytest_plugin_registered --> [] [hook]
    pytest_configure [hook]
        config: <_pytest.config.Config object at 0x7fa936766950>
      pytest_plugin_registered [hook]
          plugin: <_pytest.cacheprovider.LFPlugin object at 0x7fa935bb5b90>
          plugin_name: lfplugin
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <_pytest.cacheprovider.NFPlugin object at 0x7fa935c04150>
          plugin_name: nfplugin
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
    early skip of rewriting module: faulthandler [assertion]
      pytest_plugin_registered [hook]
          plugin: <class '_pytest.legacypath.LegacyTmpdirPlugin'>
          plugin_name: legacypath-tmpdir
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
    early skip of rewriting module: pdb [assertion]
    early skip of rewriting module: cmd [assertion]
    early skip of rewriting module: code [assertion]
    early skip of rewriting module: codeop [assertion]
      pytest_plugin_registered [hook]
          plugin: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
          plugin_name: 140364737501392
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <_pytest.config.Config object at 0x7fa936766950>
          plugin_name: pytestconfig
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.mark' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/mark/__init__.py'>
          plugin_name: mark
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.main' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/main.py'>
          plugin_name: main
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.runner' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/runner.py'>
          plugin_name: runner
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.fixtures' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/fixtures.py'>
          plugin_name: fixtures
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.helpconfig' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/helpconfig.py'>
          plugin_name: helpconfig
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.python' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/python.py'>
          plugin_name: python
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.terminal' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/terminal.py'>
          plugin_name: terminal
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.debugging' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/debugging.py'>
          plugin_name: debugging
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.unittest' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/unittest.py'>
          plugin_name: unittest
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.capture' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/capture.py'>
          plugin_name: capture
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.skipping' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/skipping.py'>
          plugin_name: skipping
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.legacypath' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/legacypath.py'>
          plugin_name: legacypath
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.tmpdir' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/tmpdir.py'>
          plugin_name: tmpdir
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.monkeypatch' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/monkeypatch.py'>
          plugin_name: monkeypatch
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.recwarn' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/recwarn.py'>
          plugin_name: recwarn
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.pastebin' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/pastebin.py'>
          plugin_name: pastebin
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.assertion' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/assertion/__init__.py'>
          plugin_name: assertion
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.junitxml' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/junitxml.py'>
          plugin_name: junitxml
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.doctest' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/doctest.py'>
          plugin_name: doctest
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.cacheprovider' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/cacheprovider.py'>
          plugin_name: cacheprovider
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.freeze_support' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/freeze_support.py'>
          plugin_name: freeze_support
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.setuponly' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/setuponly.py'>
          plugin_name: setuponly
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.setupplan' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/setupplan.py'>
          plugin_name: setupplan
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.stepwise' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/stepwise.py'>
          plugin_name: stepwise
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.warnings' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/warnings.py'>
          plugin_name: warnings
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.logging' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/logging.py'>
          plugin_name: logging
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.reports' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/reports.py'>
          plugin_name: reports
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.python_path' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/python_path.py'>
          plugin_name: python_path
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.unraisableexception' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/unraisableexception.py'>
          plugin_name: unraisableexception
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.threadexception' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/threadexception.py'>
          plugin_name: threadexception
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.faulthandler' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/faulthandler.py'>
          plugin_name: faulthandler
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <CaptureManager _method='fd' _global_capturing=<MultiCapture out=<FDCapture 1 oldfd=5 _state='suspended' tmpfile=<_io.TextIOWrapper name="<_io.FileIO name=6 mode='rb+' closefd=True>" mode='r+' encoding='utf-8'>> err=<FDCapture 2 oldfd=7 _state='suspended' tmpfile=<_io.TextIOWrapper name="<_io.FileIO name=8 mode='rb+' closefd=True>" mode='r+' encoding='utf-8'>> in_=<FDCapture 0 oldfd=3 _state='started' tmpfile=<_io.TextIOWrapper name='/dev/null' mode='r' encoding='utf-8'>> _state='suspended' _in_suspended=False> _capture_fixture=None>
          plugin_name: capturemanager
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <Session  exitstatus=<ExitCode.OK: 0> testsfailed=0 testscollected=0>
          plugin_name: session
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <_pytest.cacheprovider.LFPlugin object at 0x7fa935bb5b90>
          plugin_name: lfplugin
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <_pytest.cacheprovider.NFPlugin object at 0x7fa935c04150>
          plugin_name: nfplugin
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <class '_pytest.legacypath.LegacyTmpdirPlugin'>
          plugin_name: legacypath-tmpdir
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <_pytest.terminal.TerminalReporter object at 0x7fa935d23b90>
          plugin_name: terminalreporter
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <_pytest.logging.LoggingPlugin object at 0x7fa935c2dc10>
          plugin_name: logging-plugin
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
    finish pytest_configure --> [] [hook]
    pytest_sessionstart [hook]
        session: <Session  exitstatus=<ExitCode.OK: 0> testsfailed=0 testscollected=0>
      pytest_plugin_registered [hook]
          plugin: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
          plugin_name: 140364737501392
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <_pytest.config.Config object at 0x7fa936766950>
          plugin_name: pytestconfig
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.mark' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/mark/__init__.py'>
          plugin_name: mark
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.main' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/main.py'>
          plugin_name: main
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.runner' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/runner.py'>
          plugin_name: runner
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.fixtures' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/fixtures.py'>
          plugin_name: fixtures
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.helpconfig' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/helpconfig.py'>
          plugin_name: helpconfig
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.python' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/python.py'>
          plugin_name: python
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.terminal' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/terminal.py'>
          plugin_name: terminal
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.debugging' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/debugging.py'>
          plugin_name: debugging
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.unittest' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/unittest.py'>
          plugin_name: unittest
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.capture' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/capture.py'>
          plugin_name: capture
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.skipping' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/skipping.py'>
          plugin_name: skipping
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.legacypath' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/legacypath.py'>
          plugin_name: legacypath
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.tmpdir' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/tmpdir.py'>
          plugin_name: tmpdir
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.monkeypatch' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/monkeypatch.py'>
          plugin_name: monkeypatch
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.recwarn' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/recwarn.py'>
          plugin_name: recwarn
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.pastebin' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/pastebin.py'>
          plugin_name: pastebin
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.assertion' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/assertion/__init__.py'>
          plugin_name: assertion
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.junitxml' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/junitxml.py'>
          plugin_name: junitxml
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.doctest' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/doctest.py'>
          plugin_name: doctest
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.cacheprovider' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/cacheprovider.py'>
          plugin_name: cacheprovider
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.freeze_support' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/freeze_support.py'>
          plugin_name: freeze_support
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.setuponly' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/setuponly.py'>
          plugin_name: setuponly
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.setupplan' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/setupplan.py'>
          plugin_name: setupplan
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.stepwise' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/stepwise.py'>
          plugin_name: stepwise
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.warnings' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/warnings.py'>
          plugin_name: warnings
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.logging' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/logging.py'>
          plugin_name: logging
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.reports' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/reports.py'>
          plugin_name: reports
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.python_path' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/python_path.py'>
          plugin_name: python_path
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.unraisableexception' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/unraisableexception.py'>
          plugin_name: unraisableexception
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.threadexception' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/threadexception.py'>
          plugin_name: threadexception
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <module '_pytest.faulthandler' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/faulthandler.py'>
          plugin_name: faulthandler
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <CaptureManager _method='fd' _global_capturing=<MultiCapture out=<FDCapture 1 oldfd=5 _state='suspended' tmpfile=<_io.TextIOWrapper name="<_io.FileIO name=6 mode='rb+' closefd=True>" mode='r+' encoding='utf-8'>> err=<FDCapture 2 oldfd=7 _state='suspended' tmpfile=<_io.TextIOWrapper name="<_io.FileIO name=8 mode='rb+' closefd=True>" mode='r+' encoding='utf-8'>> in_=<FDCapture 0 oldfd=3 _state='started' tmpfile=<_io.TextIOWrapper name='/dev/null' mode='r' encoding='utf-8'>> _state='suspended' _in_suspended=False> _capture_fixture=None>
          plugin_name: capturemanager
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <Session  exitstatus=<ExitCode.OK: 0> testsfailed=0 testscollected=0>
          plugin_name: session
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <_pytest.cacheprovider.LFPlugin object at 0x7fa935bb5b90>
          plugin_name: lfplugin
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <_pytest.cacheprovider.NFPlugin object at 0x7fa935c04150>
          plugin_name: nfplugin
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <class '_pytest.legacypath.LegacyTmpdirPlugin'>
          plugin_name: legacypath-tmpdir
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <_pytest.terminal.TerminalReporter object at 0x7fa935d23b90>
          plugin_name: terminalreporter
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <_pytest.logging.LoggingPlugin object at 0x7fa935c2dc10>
          plugin_name: logging-plugin
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_plugin_registered [hook]
          plugin: <_pytest.fixtures.FixtureManager object at 0x7fa935c2e990>
          plugin_name: funcmanage
          manager: <_pytest.config.PytestPluginManager object at 0x7fa9365158d0>
      finish pytest_plugin_registered --> [] [hook]
      pytest_report_header [hook]
          config: <_pytest.config.Config object at 0x7fa936766950>
          start_path: /home/user/Documents/repos/openptv/py_bind/test
          startdir: /home/user/Documents/repos/openptv/py_bind/test
      finish pytest_report_header --> [['rootdir: /home/user/Documents/repos/openptv', 'configfile: pyproject.toml'], ['using: pytest-8.3.4']] [hook]
    finish pytest_sessionstart --> [] [hook]
    pytest_collection [hook]
        session: <Session  exitstatus=<ExitCode.OK: 0> testsfailed=0 testscollected=0>
    perform_collect <Session  exitstatus=<ExitCode.OK: 0> testsfailed=0 testscollected=0> ['/home/user/Documents/repos/openptv/py_bind/test'] [collection]
        pytest_collectstart [hook]
            collector: <Session  exitstatus=<ExitCode.OK: 0> testsfailed=0 testscollected=0>
        finish pytest_collectstart --> [] [hook]
        pytest_make_collect_report [hook]
            collector: <Session  exitstatus=<ExitCode.OK: 0> testsfailed=0 testscollected=0>
        processing argument CollectionArgument(path=PosixPath('/home/user/Documents/repos/openptv/py_bind/test'), parts=[], module_name=None) [collection]
            pytest_collect_directory [hook]
                path: /home/user/Documents/repos/openptv
                parent: <Session  exitstatus=<ExitCode.OK: 0> testsfailed=0 testscollected=0>
            finish pytest_collect_directory --> <Dir openptv> [hook]
            pytest_collectstart [hook]
                collector: <Dir openptv>
            finish pytest_collectstart --> [] [hook]
            pytest_make_collect_report [hook]
                collector: <Dir openptv>
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/.azure-pipelines
                  path: /home/user/Documents/repos/openptv/.azure-pipelines
              finish pytest_ignore_collect --> True [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/.ci_support
                  path: /home/user/Documents/repos/openptv/.ci_support
              finish pytest_ignore_collect --> True [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/.circleci
                  path: /home/user/Documents/repos/openptv/.circleci
              finish pytest_ignore_collect --> True [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/.git
                  path: /home/user/Documents/repos/openptv/.git
              finish pytest_ignore_collect --> True [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/.github
                  path: /home/user/Documents/repos/openptv/.github
              finish pytest_ignore_collect --> True [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/.gitignore
                  path: /home/user/Documents/repos/openptv/.gitignore
              finish pytest_ignore_collect --> None [hook]
              pytest_collect_file [hook]
                  parent: <Dir openptv>
                  file_path: /home/user/Documents/repos/openptv/.gitignore
                  path: /home/user/Documents/repos/openptv/.gitignore
              finish pytest_collect_file --> [] [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/.pytest_cache
                  path: /home/user/Documents/repos/openptv/.pytest_cache
              finish pytest_ignore_collect --> True [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/.scripts
                  path: /home/user/Documents/repos/openptv/.scripts
              finish pytest_ignore_collect --> True [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/.vscode
                  path: /home/user/Documents/repos/openptv/.vscode
              finish pytest_ignore_collect --> True [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/BUILD.md
                  path: /home/user/Documents/repos/openptv/BUILD.md
              finish pytest_ignore_collect --> None [hook]
              pytest_collect_file [hook]
                  parent: <Dir openptv>
                  file_path: /home/user/Documents/repos/openptv/BUILD.md
                  path: /home/user/Documents/repos/openptv/BUILD.md
              finish pytest_collect_file --> [] [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/COPYING.LESSER
                  path: /home/user/Documents/repos/openptv/COPYING.LESSER
              finish pytest_ignore_collect --> None [hook]
              pytest_collect_file [hook]
                  parent: <Dir openptv>
                  file_path: /home/user/Documents/repos/openptv/COPYING.LESSER
                  path: /home/user/Documents/repos/openptv/COPYING.LESSER
              finish pytest_collect_file --> [] [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/README.md
                  path: /home/user/Documents/repos/openptv/README.md
              finish pytest_ignore_collect --> None [hook]
              pytest_collect_file [hook]
                  parent: <Dir openptv>
                  file_path: /home/user/Documents/repos/openptv/README.md
                  path: /home/user/Documents/repos/openptv/README.md
              finish pytest_collect_file --> [] [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/azure-pipelines.yml
                  path: /home/user/Documents/repos/openptv/azure-pipelines.yml
              finish pytest_ignore_collect --> None [hook]
              pytest_collect_file [hook]
                  parent: <Dir openptv>
                  file_path: /home/user/Documents/repos/openptv/azure-pipelines.yml
                  path: /home/user/Documents/repos/openptv/azure-pipelines.yml
              finish pytest_collect_file --> [] [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/conda-forge.yml
                  path: /home/user/Documents/repos/openptv/conda-forge.yml
              finish pytest_ignore_collect --> None [hook]
              pytest_collect_file [hook]
                  parent: <Dir openptv>
                  file_path: /home/user/Documents/repos/openptv/conda-forge.yml
                  path: /home/user/Documents/repos/openptv/conda-forge.yml
              finish pytest_collect_file --> [] [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/docker
                  path: /home/user/Documents/repos/openptv/docker
              finish pytest_ignore_collect --> None [hook]
              pytest_collect_directory [hook]
                  path: /home/user/Documents/repos/openptv/docker
                  parent: <Dir openptv>
              finish pytest_collect_directory --> <Dir docker> [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/docs
                  path: /home/user/Documents/repos/openptv/docs
              finish pytest_ignore_collect --> None [hook]
              pytest_collect_directory [hook]
                  path: /home/user/Documents/repos/openptv/docs
                  parent: <Dir openptv>
              finish pytest_collect_directory --> <Dir docs> [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/liboptv
                  path: /home/user/Documents/repos/openptv/liboptv
              finish pytest_ignore_collect --> None [hook]
              pytest_collect_directory [hook]
                  path: /home/user/Documents/repos/openptv/liboptv
                  parent: <Dir openptv>
              finish pytest_collect_directory --> <Dir liboptv> [hook]
              pytest_collect_directory [hook]
                  path: /home/user/Documents/repos/openptv/py_bind
                  parent: <Dir openptv>
              finish pytest_collect_directory --> <Package py_bind> [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/pyproject.toml
                  path: /home/user/Documents/repos/openptv/pyproject.toml
              finish pytest_ignore_collect --> None [hook]
              pytest_collect_file [hook]
                  parent: <Dir openptv>
                  file_path: /home/user/Documents/repos/openptv/pyproject.toml
                  path: /home/user/Documents/repos/openptv/pyproject.toml
              finish pytest_collect_file --> [] [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/recipe
                  path: /home/user/Documents/repos/openptv/recipe
              finish pytest_ignore_collect --> None [hook]
              pytest_collect_directory [hook]
                  path: /home/user/Documents/repos/openptv/recipe
                  parent: <Dir openptv>
              finish pytest_collect_directory --> <Dir recipe> [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/viewer
                  path: /home/user/Documents/repos/openptv/viewer
              finish pytest_ignore_collect --> None [hook]
              pytest_collect_directory [hook]
                  path: /home/user/Documents/repos/openptv/viewer
                  parent: <Dir openptv>
              finish pytest_collect_directory --> <Dir viewer> [hook]
            finish pytest_make_collect_report --> <CollectReport '.' lenresult=6 outcome='passed'> [hook]
            pytest_collectstart [hook]
                collector: <Package py_bind>
            finish pytest_collectstart --> [] [hook]
            pytest_make_collect_report [hook]
                collector: <Package py_bind>
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/py_bind/__init__.py
                  path: /home/user/Documents/repos/openptv/py_bind/__init__.py
              finish pytest_ignore_collect --> None [hook]
              pytest_collect_file [hook]
                  parent: <Package py_bind>
                  file_path: /home/user/Documents/repos/openptv/py_bind/__init__.py
                  path: /home/user/Documents/repos/openptv/py_bind/__init__.py
              finish pytest_collect_file --> [] [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/py_bind/.eggs
                  path: /home/user/Documents/repos/openptv/py_bind/.eggs
              finish pytest_ignore_collect --> True [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/py_bind/.pytest_cache
                  path: /home/user/Documents/repos/openptv/py_bind/.pytest_cache
              finish pytest_ignore_collect --> True [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/py_bind/.vscode
                  path: /home/user/Documents/repos/openptv/py_bind/.vscode
              finish pytest_ignore_collect --> True [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/py_bind/README.txt
                  path: /home/user/Documents/repos/openptv/py_bind/README.txt
              finish pytest_ignore_collect --> None [hook]
              pytest_collect_file [hook]
                  parent: <Package py_bind>
                  file_path: /home/user/Documents/repos/openptv/py_bind/README.txt
                  path: /home/user/Documents/repos/openptv/py_bind/README.txt
              finish pytest_collect_file --> [] [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/py_bind/build
                  path: /home/user/Documents/repos/openptv/py_bind/build
              finish pytest_ignore_collect --> True [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/py_bind/liboptv
                  path: /home/user/Documents/repos/openptv/py_bind/liboptv
              finish pytest_ignore_collect --> None [hook]
              pytest_collect_directory [hook]
                  path: /home/user/Documents/repos/openptv/py_bind/liboptv
                  parent: <Package py_bind>
              finish pytest_collect_directory --> <Dir liboptv> [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/py_bind/optv
                  path: /home/user/Documents/repos/openptv/py_bind/optv
              finish pytest_ignore_collect --> None [hook]
              pytest_collect_directory [hook]
                  path: /home/user/Documents/repos/openptv/py_bind/optv
                  parent: <Package py_bind>
              finish pytest_collect_directory --> <Package optv> [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/py_bind/optv.egg-info
                  path: /home/user/Documents/repos/openptv/py_bind/optv.egg-info
              finish pytest_ignore_collect --> None [hook]
              pytest_collect_directory [hook]
                  path: /home/user/Documents/repos/openptv/py_bind/optv.egg-info
                  parent: <Package py_bind>
              finish pytest_collect_directory --> <Dir optv.egg-info> [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/py_bind/pyproject.toml
                  path: /home/user/Documents/repos/openptv/py_bind/pyproject.toml
              finish pytest_ignore_collect --> None [hook]
              pytest_collect_file [hook]
                  parent: <Package py_bind>
                  file_path: /home/user/Documents/repos/openptv/py_bind/pyproject.toml
                  path: /home/user/Documents/repos/openptv/py_bind/pyproject.toml
              finish pytest_collect_file --> [] [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/py_bind/requirements.txt
                  path: /home/user/Documents/repos/openptv/py_bind/requirements.txt
              finish pytest_ignore_collect --> None [hook]
              pytest_collect_file [hook]
                  parent: <Package py_bind>
                  file_path: /home/user/Documents/repos/openptv/py_bind/requirements.txt
                  path: /home/user/Documents/repos/openptv/py_bind/requirements.txt
              finish pytest_collect_file --> [] [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/py_bind/setup.py
                  path: /home/user/Documents/repos/openptv/py_bind/setup.py
              finish pytest_ignore_collect --> None [hook]
              pytest_collect_file [hook]
                  parent: <Package py_bind>
                  file_path: /home/user/Documents/repos/openptv/py_bind/setup.py
                  path: /home/user/Documents/repos/openptv/py_bind/setup.py
              finish pytest_collect_file --> [] [hook]
              pytest_collect_directory [hook]
                  path: /home/user/Documents/repos/openptv/py_bind/test
                  parent: <Package py_bind>
              finish pytest_collect_directory --> <Dir test> [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/py_bind/testing_fodder
                  path: /home/user/Documents/repos/openptv/py_bind/testing_fodder
              finish pytest_ignore_collect --> None [hook]
              pytest_collect_directory [hook]
                  path: /home/user/Documents/repos/openptv/py_bind/testing_fodder
                  parent: <Package py_bind>
              finish pytest_collect_directory --> <Dir testing_fodder> [hook]
              pytest_ignore_collect [hook]
                  config: <_pytest.config.Config object at 0x7fa936766950>
                  collection_path: /home/user/Documents/repos/openptv/py_bind/wheelhouse
                  path: /home/user/Documents/repos/openptv/py_bind/wheelhouse
              finish pytest_ignore_collect --> None [hook]
              pytest_collect_directory [hook]
                  path: /home/user/Documents/repos/openptv/py_bind/wheelhouse
                  parent: <Package py_bind>
              finish pytest_collect_directory --> <Dir wheelhouse> [hook]
            finish pytest_make_collect_report --> <CollectReport 'py_bind' lenresult=6 outcome='passed'> [hook]
        finish pytest_make_collect_report --> <CollectReport '' lenresult=1 outcome='passed'> [hook]
        pytest_collectreport [hook]
            report: <CollectReport '' lenresult=1 outcome='passed'>
        finish pytest_collectreport --> [] [hook]
    genitems <Dir test> [collection]
      pytest_collectstart [hook]
          collector: <Dir test>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Dir test>
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/__pycache__
            path: /home/user/Documents/repos/openptv/py_bind/test/__pycache__
        finish pytest_ignore_collect --> True [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/test_burgers.py
            path: /home/user/Documents/repos/openptv/py_bind/test/test_burgers.py
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir test>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/test_burgers.py
            path: /home/user/Documents/repos/openptv/py_bind/test/test_burgers.py
          pytest_pycollect_makemodule [hook]
              parent: <Dir test>
              module_path: /home/user/Documents/repos/openptv/py_bind/test/test_burgers.py
              path: /home/user/Documents/repos/openptv/py_bind/test/test_burgers.py
          finish pytest_pycollect_makemodule --> <Module test_burgers.py> [hook]
        finish pytest_collect_file --> [<Module test_burgers.py>] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/test_calibration_binding.py
            path: /home/user/Documents/repos/openptv/py_bind/test/test_calibration_binding.py
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir test>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/test_calibration_binding.py
            path: /home/user/Documents/repos/openptv/py_bind/test/test_calibration_binding.py
          pytest_pycollect_makemodule [hook]
              parent: <Dir test>
              module_path: /home/user/Documents/repos/openptv/py_bind/test/test_calibration_binding.py
              path: /home/user/Documents/repos/openptv/py_bind/test/test_calibration_binding.py
          finish pytest_pycollect_makemodule --> <Module test_calibration_binding.py> [hook]
        finish pytest_collect_file --> [<Module test_calibration_binding.py>] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/test_corresp.py
            path: /home/user/Documents/repos/openptv/py_bind/test/test_corresp.py
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir test>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/test_corresp.py
            path: /home/user/Documents/repos/openptv/py_bind/test/test_corresp.py
          pytest_pycollect_makemodule [hook]
              parent: <Dir test>
              module_path: /home/user/Documents/repos/openptv/py_bind/test/test_corresp.py
              path: /home/user/Documents/repos/openptv/py_bind/test/test_corresp.py
          finish pytest_pycollect_makemodule --> <Module test_corresp.py> [hook]
        finish pytest_collect_file --> [<Module test_corresp.py>] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/test_epipolar.py
            path: /home/user/Documents/repos/openptv/py_bind/test/test_epipolar.py
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir test>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/test_epipolar.py
            path: /home/user/Documents/repos/openptv/py_bind/test/test_epipolar.py
          pytest_pycollect_makemodule [hook]
              parent: <Dir test>
              module_path: /home/user/Documents/repos/openptv/py_bind/test/test_epipolar.py
              path: /home/user/Documents/repos/openptv/py_bind/test/test_epipolar.py
          finish pytest_pycollect_makemodule --> <Module test_epipolar.py> [hook]
        finish pytest_collect_file --> [<Module test_epipolar.py>] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/test_framebuf.py
            path: /home/user/Documents/repos/openptv/py_bind/test/test_framebuf.py
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir test>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/test_framebuf.py
            path: /home/user/Documents/repos/openptv/py_bind/test/test_framebuf.py
          pytest_pycollect_makemodule [hook]
              parent: <Dir test>
              module_path: /home/user/Documents/repos/openptv/py_bind/test/test_framebuf.py
              path: /home/user/Documents/repos/openptv/py_bind/test/test_framebuf.py
          finish pytest_pycollect_makemodule --> <Module test_framebuf.py> [hook]
        finish pytest_collect_file --> [<Module test_framebuf.py>] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/test_image_processing.py
            path: /home/user/Documents/repos/openptv/py_bind/test/test_image_processing.py
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir test>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/test_image_processing.py
            path: /home/user/Documents/repos/openptv/py_bind/test/test_image_processing.py
          pytest_pycollect_makemodule [hook]
              parent: <Dir test>
              module_path: /home/user/Documents/repos/openptv/py_bind/test/test_image_processing.py
              path: /home/user/Documents/repos/openptv/py_bind/test/test_image_processing.py
          finish pytest_pycollect_makemodule --> <Module test_image_processing.py> [hook]
        finish pytest_collect_file --> [<Module test_image_processing.py>] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/test_img_coord.py
            path: /home/user/Documents/repos/openptv/py_bind/test/test_img_coord.py
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir test>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/test_img_coord.py
            path: /home/user/Documents/repos/openptv/py_bind/test/test_img_coord.py
          pytest_pycollect_makemodule [hook]
              parent: <Dir test>
              module_path: /home/user/Documents/repos/openptv/py_bind/test/test_img_coord.py
              path: /home/user/Documents/repos/openptv/py_bind/test/test_img_coord.py
          finish pytest_pycollect_makemodule --> <Module test_img_coord.py> [hook]
        finish pytest_collect_file --> [<Module test_img_coord.py>] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/test_orientation.py
            path: /home/user/Documents/repos/openptv/py_bind/test/test_orientation.py
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir test>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/test_orientation.py
            path: /home/user/Documents/repos/openptv/py_bind/test/test_orientation.py
          pytest_pycollect_makemodule [hook]
              parent: <Dir test>
              module_path: /home/user/Documents/repos/openptv/py_bind/test/test_orientation.py
              path: /home/user/Documents/repos/openptv/py_bind/test/test_orientation.py
          finish pytest_pycollect_makemodule --> <Module test_orientation.py> [hook]
        finish pytest_collect_file --> [<Module test_orientation.py>] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/test_parameters_bindings.py
            path: /home/user/Documents/repos/openptv/py_bind/test/test_parameters_bindings.py
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir test>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/test_parameters_bindings.py
            path: /home/user/Documents/repos/openptv/py_bind/test/test_parameters_bindings.py
          pytest_pycollect_makemodule [hook]
              parent: <Dir test>
              module_path: /home/user/Documents/repos/openptv/py_bind/test/test_parameters_bindings.py
              path: /home/user/Documents/repos/openptv/py_bind/test/test_parameters_bindings.py
          finish pytest_pycollect_makemodule --> <Module test_parameters_bindings.py> [hook]
        finish pytest_collect_file --> [<Module test_parameters_bindings.py>] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/test_segmentation.py
            path: /home/user/Documents/repos/openptv/py_bind/test/test_segmentation.py
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir test>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/test_segmentation.py
            path: /home/user/Documents/repos/openptv/py_bind/test/test_segmentation.py
          pytest_pycollect_makemodule [hook]
              parent: <Dir test>
              module_path: /home/user/Documents/repos/openptv/py_bind/test/test_segmentation.py
              path: /home/user/Documents/repos/openptv/py_bind/test/test_segmentation.py
          finish pytest_pycollect_makemodule --> <Module test_segmentation.py> [hook]
        finish pytest_collect_file --> [<Module test_segmentation.py>] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/test_tracker.py
            path: /home/user/Documents/repos/openptv/py_bind/test/test_tracker.py
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir test>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/test_tracker.py
            path: /home/user/Documents/repos/openptv/py_bind/test/test_tracker.py
          pytest_pycollect_makemodule [hook]
              parent: <Dir test>
              module_path: /home/user/Documents/repos/openptv/py_bind/test/test_tracker.py
              path: /home/user/Documents/repos/openptv/py_bind/test/test_tracker.py
          finish pytest_pycollect_makemodule --> <Module test_tracker.py> [hook]
        finish pytest_collect_file --> [<Module test_tracker.py>] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/test_trafo_bindings.py
            path: /home/user/Documents/repos/openptv/py_bind/test/test_trafo_bindings.py
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir test>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/test_trafo_bindings.py
            path: /home/user/Documents/repos/openptv/py_bind/test/test_trafo_bindings.py
          pytest_pycollect_makemodule [hook]
              parent: <Dir test>
              module_path: /home/user/Documents/repos/openptv/py_bind/test/test_trafo_bindings.py
              path: /home/user/Documents/repos/openptv/py_bind/test/test_trafo_bindings.py
          finish pytest_pycollect_makemodule --> <Module test_trafo_bindings.py> [hook]
        finish pytest_collect_file --> [<Module test_trafo_bindings.py>] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_directory [hook]
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder
            parent: <Dir test>
        finish pytest_collect_directory --> <Dir testing_fodder> [hook]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test' lenresult=13 outcome='passed'> [hook]
    genitems <Module test_burgers.py> [collection]
      pytest_collectstart [hook]
          collector: <Module test_burgers.py>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Module test_burgers.py>
      find_module called for: test_burgers [assertion]
      matched test file '/home/user/Documents/repos/openptv/py_bind/test/test_burgers.py' [assertion]
      found cached rewritten pyc for /home/user/Documents/repos/openptv/py_bind/test/test_burgers.py [assertion]
      early skip of rewriting module: yaml [assertion]
      early skip of rewriting module: yaml.error [assertion]
      early skip of rewriting module: yaml.tokens [assertion]
      early skip of rewriting module: yaml.events [assertion]
      early skip of rewriting module: yaml.nodes [assertion]
      early skip of rewriting module: yaml.loader [assertion]
      early skip of rewriting module: yaml.reader [assertion]
      early skip of rewriting module: yaml.scanner [assertion]
      early skip of rewriting module: yaml.parser [assertion]
      early skip of rewriting module: yaml.composer [assertion]
      early skip of rewriting module: yaml.constructor [assertion]
      early skip of rewriting module: yaml.resolver [assertion]
      early skip of rewriting module: yaml.dumper [assertion]
      early skip of rewriting module: yaml.emitter [assertion]
      early skip of rewriting module: yaml.serializer [assertion]
      early skip of rewriting module: yaml.representer [assertion]
      early skip of rewriting module: yaml.cyaml [assertion]
      early skip of rewriting module: yaml._yaml [assertion]
      early skip of rewriting module: optv [assertion]
      early skip of rewriting module: optv.tracker [assertion]
      early skip of rewriting module: optv.parameters [assertion]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/test_burgers.py' lenresult=0 outcome='failed'> [hook]
      pytest_exception_interact [hook]
          node: <Module test_burgers.py>
          call: <CallInfo when='collect' excinfo=<ExceptionInfo CollectError("ImportError while importing test module '/home/user/Documents/repos/openptv/py_bind/test/test_burgers.py...r/Documents/repos/openptv/py_bind/optv/parameters.cpython-311-x86_64-linux-gnu.so: undefined symbol: c_read_track_par") tblen=7>>
          report: <CollectReport 'py_bind/test/test_burgers.py' lenresult=0 outcome='failed'>
      finish pytest_exception_interact --> [] [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/test_burgers.py' lenresult=0 outcome='failed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Module test_calibration_binding.py> [collection]
      pytest_collectstart [hook]
          collector: <Module test_calibration_binding.py>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Module test_calibration_binding.py>
      find_module called for: test_calibration_binding [assertion]
      matched test file '/home/user/Documents/repos/openptv/py_bind/test/test_calibration_binding.py' [assertion]
      found cached rewritten pyc for /home/user/Documents/repos/openptv/py_bind/test/test_calibration_binding.py [assertion]
      early skip of rewriting module: optv.calibration [assertion]
      early skip of rewriting module: numpy [assertion]
      early skip of rewriting module: numpy._globals [assertion]
      early skip of rewriting module: numpy._utils [assertion]
      early skip of rewriting module: numpy._utils._convertions [assertion]
      early skip of rewriting module: numpy._expired_attrs_2_0 [assertion]
      early skip of rewriting module: numpy.version [assertion]
      early skip of rewriting module: numpy._distributor_init [assertion]
      early skip of rewriting module: mkl [assertion]
      early skip of rewriting module: ctypes [assertion]
      early skip of rewriting module: _ctypes [assertion]
      early skip of rewriting module: ctypes._endian [assertion]
      early skip of rewriting module: mkl._mklinit [assertion]
      early skip of rewriting module: mkl._py_mkl_service [assertion]
      early skip of rewriting module: numpy.__config__ [assertion]
      early skip of rewriting module: numpy._core [assertion]
      early skip of rewriting module: numpy._core.multiarray [assertion]
      early skip of rewriting module: numpy._core.overrides [assertion]
      early skip of rewriting module: numpy._utils._inspect [assertion]
      early skip of rewriting module: numpy._core._multiarray_umath [assertion]
      early skip of rewriting module: numpy.exceptions [assertion]
      early skip of rewriting module: numpy._core._exceptions [assertion]
      early skip of rewriting module: numpy._core.printoptions [assertion]
      early skip of rewriting module: contextvars [assertion]
      early skip of rewriting module: _contextvars [assertion]
      early skip of rewriting module: numpy.dtypes [assertion]
      early skip of rewriting module: numpy._core.umath [assertion]
      early skip of rewriting module: numpy._core.numerictypes [assertion]
      early skip of rewriting module: numpy._core._string_helpers [assertion]
      early skip of rewriting module: numpy._core._type_aliases [assertion]
      early skip of rewriting module: numpy._core._dtype [assertion]
      early skip of rewriting module: numpy._core.numeric [assertion]
      early skip of rewriting module: numpy._core.shape_base [assertion]
      early skip of rewriting module: numpy._core.fromnumeric [assertion]
      early skip of rewriting module: numpy._core._methods [assertion]
      early skip of rewriting module: pickle [assertion]
      early skip of rewriting module: _compat_pickle [assertion]
      early skip of rewriting module: _pickle [assertion]
      early skip of rewriting module: org [assertion]
      early skip of rewriting module: numpy._core._ufunc_config [assertion]
      early skip of rewriting module: numpy._core.arrayprint [assertion]
      early skip of rewriting module: numpy._core._asarray [assertion]
      early skip of rewriting module: numpy._core.records [assertion]
      early skip of rewriting module: numpy._core.memmap [assertion]
      early skip of rewriting module: numpy._core.function_base [assertion]
      early skip of rewriting module: numpy._core._machar [assertion]
      early skip of rewriting module: numpy._core.getlimits [assertion]
      early skip of rewriting module: numpy._core.einsumfunc [assertion]
      early skip of rewriting module: numpy._core._add_newdocs [assertion]
      early skip of rewriting module: numpy._core._add_newdocs_scalars [assertion]
      early skip of rewriting module: numpy._core._dtype_ctypes [assertion]
      early skip of rewriting module: numpy._core._internal [assertion]
      early skip of rewriting module: numpy._pytesttester [assertion]
      early skip of rewriting module: numpy.lib [assertion]
      early skip of rewriting module: numpy.lib.array_utils [assertion]
      early skip of rewriting module: numpy.lib._array_utils_impl [assertion]
      early skip of rewriting module: numpy.lib.introspect [assertion]
      early skip of rewriting module: numpy.lib.mixins [assertion]
      early skip of rewriting module: numpy.lib.npyio [assertion]
      early skip of rewriting module: numpy.lib._npyio_impl [assertion]
      early skip of rewriting module: numpy.lib.format [assertion]
      early skip of rewriting module: numpy.lib._utils_impl [assertion]
      early skip of rewriting module: numpy.lib._datasource [assertion]
      early skip of rewriting module: numpy.lib._iotools [assertion]
      early skip of rewriting module: numpy.lib.scimath [assertion]
      early skip of rewriting module: numpy.lib._scimath_impl [assertion]
      early skip of rewriting module: numpy.lib._type_check_impl [assertion]
      early skip of rewriting module: numpy.lib._ufunclike_impl [assertion]
      early skip of rewriting module: numpy.lib.stride_tricks [assertion]
      early skip of rewriting module: numpy.lib._stride_tricks_impl [assertion]
      early skip of rewriting module: numpy.lib._index_tricks_impl [assertion]
      early skip of rewriting module: numpy.matrixlib [assertion]
      early skip of rewriting module: numpy.matrixlib.defmatrix [assertion]
      early skip of rewriting module: numpy.linalg [assertion]
      early skip of rewriting module: numpy.linalg.linalg [assertion]
      early skip of rewriting module: numpy.linalg._linalg [assertion]
      early skip of rewriting module: numpy.lib._twodim_base_impl [assertion]
      early skip of rewriting module: numpy.linalg._umath_linalg [assertion]
      early skip of rewriting module: numpy._typing [assertion]
      early skip of rewriting module: numpy._typing._nested_sequence [assertion]
      early skip of rewriting module: numpy._typing._nbit_base [assertion]
      early skip of rewriting module: numpy._typing._nbit [assertion]
      early skip of rewriting module: numpy._typing._char_codes [assertion]
      early skip of rewriting module: numpy._typing._scalars [assertion]
      early skip of rewriting module: numpy._typing._shape [assertion]
      early skip of rewriting module: numpy._typing._dtype_like [assertion]
      early skip of rewriting module: numpy._typing._array_like [assertion]
      early skip of rewriting module: numpy._typing._ufunc [assertion]
      early skip of rewriting module: numpy.lib._function_base_impl [assertion]
      early skip of rewriting module: numpy.lib._histograms_impl [assertion]
      early skip of rewriting module: numpy.lib._nanfunctions_impl [assertion]
      early skip of rewriting module: numpy.lib._shape_base_impl [assertion]
      early skip of rewriting module: numpy.lib._arraysetops_impl [assertion]
      early skip of rewriting module: numpy.lib._polynomial_impl [assertion]
      early skip of rewriting module: numpy.lib._arrayterator_impl [assertion]
      early skip of rewriting module: numpy.lib._arraypad_impl [assertion]
      early skip of rewriting module: numpy.lib._version [assertion]
      early skip of rewriting module: numpy._array_api_info [assertion]
      early skip of rewriting module: optv.numpy [assertion]
      early skip of rewriting module: filecmp [assertion]
        pytest_pycollect_makeitem [hook]
            collector: <Module test_calibration_binding.py>
            name: @py_builtins
            obj: <module 'builtins' (built-in)>
        finish pytest_pycollect_makeitem --> None [hook]
        pytest_pycollect_makeitem [hook]
            collector: <Module test_calibration_binding.py>
            name: @pytest_ar
            obj: <module '_pytest.assertion.rewrite' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/assertion/rewrite.py'>
        finish pytest_pycollect_makeitem --> None [hook]
        pytest_pycollect_makeitem [hook]
            collector: <Module test_calibration_binding.py>
            name: unittest
            obj: <module 'unittest' from '/home/user/miniconda3/envs/openptv/lib/python3.11/unittest/__init__.py'>
        finish pytest_pycollect_makeitem --> None [hook]
        pytest_pycollect_makeitem [hook]
            collector: <Module test_calibration_binding.py>
            name: Calibration
            obj: <class 'py_bind.optv.calibration.Calibration'>
        finish pytest_pycollect_makeitem --> None [hook]
        pytest_pycollect_makeitem [hook]
            collector: <Module test_calibration_binding.py>
            name: numpy
            obj: <module 'numpy' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/numpy/__init__.py'>
        finish pytest_pycollect_makeitem --> None [hook]
        pytest_pycollect_makeitem [hook]
            collector: <Module test_calibration_binding.py>
            name: os
            obj: <module 'os' (frozen)>
        finish pytest_pycollect_makeitem --> None [hook]
        pytest_pycollect_makeitem [hook]
            collector: <Module test_calibration_binding.py>
            name: filecmp
            obj: <module 'filecmp' from '/home/user/miniconda3/envs/openptv/lib/python3.11/filecmp.py'>
        finish pytest_pycollect_makeitem --> None [hook]
        pytest_pycollect_makeitem [hook]
            collector: <Module test_calibration_binding.py>
            name: shutil
            obj: <module 'shutil' from '/home/user/miniconda3/envs/openptv/lib/python3.11/shutil.py'>
        finish pytest_pycollect_makeitem --> None [hook]
        pytest_pycollect_makeitem [hook]
            collector: <Module test_calibration_binding.py>
            name: Test_Calibration
            obj: <class 'test_calibration_binding.Test_Calibration'>
        finish pytest_pycollect_makeitem --> <UnitTestCase Test_Calibration> [hook]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/test_calibration_binding.py' lenresult=1 outcome='passed'> [hook]
    genitems <UnitTestCase Test_Calibration> [collection]
      pytest_collectstart [hook]
          collector: <UnitTestCase Test_Calibration>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <UnitTestCase Test_Calibration>
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/test_calibration_binding.py::Test_Calibration' lenresult=8 outcome='passed'> [hook]
    genitems <TestCaseFunction test_Calibration_instantiation> [collection]
      pytest_itemcollected [hook]
          item: <TestCaseFunction test_Calibration_instantiation>
      finish pytest_itemcollected --> [] [hook]
    genitems <TestCaseFunction test_full_instantiate> [collection]
      pytest_itemcollected [hook]
          item: <TestCaseFunction test_full_instantiate>
      finish pytest_itemcollected --> [] [hook]
    genitems <TestCaseFunction test_set_angles> [collection]
      pytest_itemcollected [hook]
          item: <TestCaseFunction test_set_angles>
      finish pytest_itemcollected --> [] [hook]
    genitems <TestCaseFunction test_set_decentering> [collection]
      pytest_itemcollected [hook]
          item: <TestCaseFunction test_set_decentering>
      finish pytest_itemcollected --> [] [hook]
    genitems <TestCaseFunction test_set_glass> [collection]
      pytest_itemcollected [hook]
          item: <TestCaseFunction test_set_glass>
      finish pytest_itemcollected --> [] [hook]
    genitems <TestCaseFunction test_set_pos> [collection]
      pytest_itemcollected [hook]
          item: <TestCaseFunction test_set_pos>
      finish pytest_itemcollected --> [] [hook]
    genitems <TestCaseFunction test_set_primary> [collection]
      pytest_itemcollected [hook]
          item: <TestCaseFunction test_set_primary>
      finish pytest_itemcollected --> [] [hook]
    genitems <TestCaseFunction test_set_radial> [collection]
      pytest_itemcollected [hook]
          item: <TestCaseFunction test_set_radial>
      finish pytest_itemcollected --> [] [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/test_calibration_binding.py::Test_Calibration' lenresult=8 outcome='passed'>
      finish pytest_collectreport --> [] [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/test_calibration_binding.py' lenresult=1 outcome='passed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Module test_corresp.py> [collection]
      pytest_collectstart [hook]
          collector: <Module test_corresp.py>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Module test_corresp.py>
      find_module called for: test_corresp [assertion]
      matched test file '/home/user/Documents/repos/openptv/py_bind/test/test_corresp.py' [assertion]
      _read_pyc(/home/user/Documents/repos/openptv/py_bind/test/test_corresp.py): out of date [assertion]
      rewriting PosixPath('/home/user/Documents/repos/openptv/py_bind/test/test_corresp.py') [assertion]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/test_corresp.py' lenresult=0 outcome='failed'> [hook]
      pytest_exception_interact [hook]
          node: <Module test_corresp.py>
          call: <CallInfo when='collect' excinfo=<ExceptionInfo CollectError(ExceptionChainRepr(reprtraceback=ReprTraceback(reprentries=[ReprEntry(lines=['    mod = import_path('], r...line 1\n    versions pytest-8.3.4, python-3.11.11.final.0\n             ^^^^^^\nSyntaxError: invalid syntax'), None)])) tblen=7>>
          report: <CollectReport 'py_bind/test/test_corresp.py' lenresult=0 outcome='failed'>
      finish pytest_exception_interact --> [] [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/test_corresp.py' lenresult=0 outcome='failed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Module test_epipolar.py> [collection]
      pytest_collectstart [hook]
          collector: <Module test_epipolar.py>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Module test_epipolar.py>
      find_module called for: test_epipolar [assertion]
      matched test file '/home/user/Documents/repos/openptv/py_bind/test/test_epipolar.py' [assertion]
      found cached rewritten pyc for /home/user/Documents/repos/openptv/py_bind/test/test_epipolar.py [assertion]
      early skip of rewriting module: optv.parameters [assertion]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/test_epipolar.py' lenresult=0 outcome='failed'> [hook]
      pytest_exception_interact [hook]
          node: <Module test_epipolar.py>
          call: <CallInfo when='collect' excinfo=<ExceptionInfo CollectError("ImportError while importing test module '/home/user/Documents/repos/openptv/py_bind/test/test_epipolar.p...r/Documents/repos/openptv/py_bind/optv/parameters.cpython-311-x86_64-linux-gnu.so: undefined symbol: c_read_track_par") tblen=7>>
          report: <CollectReport 'py_bind/test/test_epipolar.py' lenresult=0 outcome='failed'>
      finish pytest_exception_interact --> [] [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/test_epipolar.py' lenresult=0 outcome='failed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Module test_framebuf.py> [collection]
      pytest_collectstart [hook]
          collector: <Module test_framebuf.py>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Module test_framebuf.py>
      find_module called for: test_framebuf [assertion]
      matched test file '/home/user/Documents/repos/openptv/py_bind/test/test_framebuf.py' [assertion]
      found cached rewritten pyc for /home/user/Documents/repos/openptv/py_bind/test/test_framebuf.py [assertion]
      early skip of rewriting module: optv.tracking_framebuf [assertion]
      early skip of rewriting module: optv.numpy [assertion]
        pytest_pycollect_makeitem [hook]
            collector: <Module test_framebuf.py>
            name: @py_builtins
            obj: <module 'builtins' (built-in)>
        finish pytest_pycollect_makeitem --> None [hook]
        pytest_pycollect_makeitem [hook]
            collector: <Module test_framebuf.py>
            name: @pytest_ar
            obj: <module '_pytest.assertion.rewrite' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/_pytest/assertion/rewrite.py'>
        finish pytest_pycollect_makeitem --> None [hook]
        pytest_pycollect_makeitem [hook]
            collector: <Module test_framebuf.py>
            name: unittest
            obj: <module 'unittest' from '/home/user/miniconda3/envs/openptv/lib/python3.11/unittest/__init__.py'>
        finish pytest_pycollect_makeitem --> None [hook]
        pytest_pycollect_makeitem [hook]
            collector: <Module test_framebuf.py>
            name: os
            obj: <module 'os' (frozen)>
        finish pytest_pycollect_makeitem --> None [hook]
        pytest_pycollect_makeitem [hook]
            collector: <Module test_framebuf.py>
            name: np
            obj: <module 'numpy' from '/home/user/miniconda3/envs/openptv/lib/python3.11/site-packages/numpy/__init__.py'>
        finish pytest_pycollect_makeitem --> None [hook]
        pytest_pycollect_makeitem [hook]
            collector: <Module test_framebuf.py>
            name: read_targets
            obj: <cyfunction read_targets at 0x7fa935f80fb0>
        finish pytest_pycollect_makeitem --> None [hook]
        pytest_pycollect_makeitem [hook]
            collector: <Module test_framebuf.py>
            name: Target
            obj: <class 'py_bind.optv.tracking_framebuf.Target'>
        finish pytest_pycollect_makeitem --> None [hook]
        pytest_pycollect_makeitem [hook]
            collector: <Module test_framebuf.py>
            name: TargetArray
            obj: <class 'py_bind.optv.tracking_framebuf.TargetArray'>
        finish pytest_pycollect_makeitem --> None [hook]
        pytest_pycollect_makeitem [hook]
            collector: <Module test_framebuf.py>
            name: Frame
            obj: <class 'py_bind.optv.tracking_framebuf.Frame'>
        finish pytest_pycollect_makeitem --> None [hook]
        pytest_pycollect_makeitem [hook]
            collector: <Module test_framebuf.py>
            name: TestTargets
            obj: <class 'test_framebuf.TestTargets'>
        finish pytest_pycollect_makeitem --> <UnitTestCase TestTargets> [hook]
        pytest_pycollect_makeitem [hook]
            collector: <Module test_framebuf.py>
            name: TestFrame
            obj: <class 'test_framebuf.TestFrame'>
        finish pytest_pycollect_makeitem --> <UnitTestCase TestFrame> [hook]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/test_framebuf.py' lenresult=2 outcome='passed'> [hook]
    genitems <UnitTestCase TestTargets> [collection]
      pytest_collectstart [hook]
          collector: <UnitTestCase TestTargets>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <UnitTestCase TestTargets>
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/test_framebuf.py::TestTargets' lenresult=5 outcome='passed'> [hook]
    genitems <TestCaseFunction test_fill_target> [collection]
      pytest_itemcollected [hook]
          item: <TestCaseFunction test_fill_target>
      finish pytest_itemcollected --> [] [hook]
    genitems <TestCaseFunction test_fill_target_array> [collection]
      pytest_itemcollected [hook]
          item: <TestCaseFunction test_fill_target_array>
      finish pytest_itemcollected --> [] [hook]
    genitems <TestCaseFunction test_read_targets> [collection]
      pytest_itemcollected [hook]
          item: <TestCaseFunction test_read_targets>
      finish pytest_itemcollected --> [] [hook]
    genitems <TestCaseFunction test_sort_y> [collection]
      pytest_itemcollected [hook]
          item: <TestCaseFunction test_sort_y>
      finish pytest_itemcollected --> [] [hook]
    genitems <TestCaseFunction test_write_targets> [collection]
      pytest_itemcollected [hook]
          item: <TestCaseFunction test_write_targets>
      finish pytest_itemcollected --> [] [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/test_framebuf.py::TestTargets' lenresult=5 outcome='passed'>
      finish pytest_collectreport --> [] [hook]
    genitems <UnitTestCase TestFrame> [collection]
      pytest_collectstart [hook]
          collector: <UnitTestCase TestFrame>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <UnitTestCase TestFrame>
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/test_framebuf.py::TestFrame' lenresult=1 outcome='passed'> [hook]
    genitems <TestCaseFunction test_read_frame> [collection]
      pytest_itemcollected [hook]
          item: <TestCaseFunction test_read_frame>
      finish pytest_itemcollected --> [] [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/test_framebuf.py::TestFrame' lenresult=1 outcome='passed'>
      finish pytest_collectreport --> [] [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/test_framebuf.py' lenresult=2 outcome='passed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Module test_image_processing.py> [collection]
      pytest_collectstart [hook]
          collector: <Module test_image_processing.py>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Module test_image_processing.py>
      find_module called for: test_image_processing [assertion]
      matched test file '/home/user/Documents/repos/openptv/py_bind/test/test_image_processing.py' [assertion]
      found cached rewritten pyc for /home/user/Documents/repos/openptv/py_bind/test/test_image_processing.py [assertion]
      early skip of rewriting module: optv.parameters [assertion]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/test_image_processing.py' lenresult=0 outcome='failed'> [hook]
      pytest_exception_interact [hook]
          node: <Module test_image_processing.py>
          call: <CallInfo when='collect' excinfo=<ExceptionInfo CollectError("ImportError while importing test module '/home/user/Documents/repos/openptv/py_bind/test/test_image_proc...r/Documents/repos/openptv/py_bind/optv/parameters.cpython-311-x86_64-linux-gnu.so: undefined symbol: c_read_track_par") tblen=7>>
          report: <CollectReport 'py_bind/test/test_image_processing.py' lenresult=0 outcome='failed'>
      finish pytest_exception_interact --> [] [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/test_image_processing.py' lenresult=0 outcome='failed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Module test_img_coord.py> [collection]
      pytest_collectstart [hook]
          collector: <Module test_img_coord.py>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Module test_img_coord.py>
      find_module called for: test_img_coord [assertion]
      matched test file '/home/user/Documents/repos/openptv/py_bind/test/test_img_coord.py' [assertion]
      found cached rewritten pyc for /home/user/Documents/repos/openptv/py_bind/test/test_img_coord.py [assertion]
      early skip of rewriting module: optv.imgcoord [assertion]
      early skip of rewriting module: optv.parameters [assertion]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/test_img_coord.py' lenresult=0 outcome='failed'> [hook]
      pytest_exception_interact [hook]
          node: <Module test_img_coord.py>
          call: <CallInfo when='collect' excinfo=<ExceptionInfo CollectError("ImportError while importing test module '/home/user/Documents/repos/openptv/py_bind/test/test_img_coord....r/Documents/repos/openptv/py_bind/optv/parameters.cpython-311-x86_64-linux-gnu.so: undefined symbol: c_read_track_par") tblen=7>>
          report: <CollectReport 'py_bind/test/test_img_coord.py' lenresult=0 outcome='failed'>
      finish pytest_exception_interact --> [] [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/test_img_coord.py' lenresult=0 outcome='failed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Module test_orientation.py> [collection]
      pytest_collectstart [hook]
          collector: <Module test_orientation.py>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Module test_orientation.py>
      find_module called for: test_orientation [assertion]
      matched test file '/home/user/Documents/repos/openptv/py_bind/test/test_orientation.py' [assertion]
      found cached rewritten pyc for /home/user/Documents/repos/openptv/py_bind/test/test_orientation.py [assertion]
      early skip of rewriting module: optv.imgcoord [assertion]
      early skip of rewriting module: optv.parameters [assertion]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/test_orientation.py' lenresult=0 outcome='failed'> [hook]
      pytest_exception_interact [hook]
          node: <Module test_orientation.py>
          call: <CallInfo when='collect' excinfo=<ExceptionInfo CollectError("ImportError while importing test module '/home/user/Documents/repos/openptv/py_bind/test/test_orientatio...r/Documents/repos/openptv/py_bind/optv/parameters.cpython-311-x86_64-linux-gnu.so: undefined symbol: c_read_track_par") tblen=7>>
          report: <CollectReport 'py_bind/test/test_orientation.py' lenresult=0 outcome='failed'>
      finish pytest_exception_interact --> [] [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/test_orientation.py' lenresult=0 outcome='failed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Module test_parameters_bindings.py> [collection]
      pytest_collectstart [hook]
          collector: <Module test_parameters_bindings.py>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Module test_parameters_bindings.py>
      find_module called for: test_parameters_bindings [assertion]
      matched test file '/home/user/Documents/repos/openptv/py_bind/test/test_parameters_bindings.py' [assertion]
      found cached rewritten pyc for /home/user/Documents/repos/openptv/py_bind/test/test_parameters_bindings.py [assertion]
      early skip of rewriting module: optv.parameters [assertion]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/test_parameters_bindings.py' lenresult=0 outcome='failed'> [hook]
      pytest_exception_interact [hook]
          node: <Module test_parameters_bindings.py>
          call: <CallInfo when='collect' excinfo=<ExceptionInfo CollectError("ImportError while importing test module '/home/user/Documents/repos/openptv/py_bind/test/test_parameters...r/Documents/repos/openptv/py_bind/optv/parameters.cpython-311-x86_64-linux-gnu.so: undefined symbol: c_read_track_par") tblen=7>>
          report: <CollectReport 'py_bind/test/test_parameters_bindings.py' lenresult=0 outcome='failed'>
      finish pytest_exception_interact --> [] [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/test_parameters_bindings.py' lenresult=0 outcome='failed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Module test_segmentation.py> [collection]
      pytest_collectstart [hook]
          collector: <Module test_segmentation.py>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Module test_segmentation.py>
      find_module called for: test_segmentation [assertion]
      matched test file '/home/user/Documents/repos/openptv/py_bind/test/test_segmentation.py' [assertion]
      found cached rewritten pyc for /home/user/Documents/repos/openptv/py_bind/test/test_segmentation.py [assertion]
      early skip of rewriting module: optv.segmentation [assertion]
      early skip of rewriting module: optv.parameters [assertion]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/test_segmentation.py' lenresult=0 outcome='failed'> [hook]
      pytest_exception_interact [hook]
          node: <Module test_segmentation.py>
          call: <CallInfo when='collect' excinfo=<ExceptionInfo CollectError("ImportError while importing test module '/home/user/Documents/repos/openptv/py_bind/test/test_segmentati...r/Documents/repos/openptv/py_bind/optv/parameters.cpython-311-x86_64-linux-gnu.so: undefined symbol: c_read_track_par") tblen=7>>
          report: <CollectReport 'py_bind/test/test_segmentation.py' lenresult=0 outcome='failed'>
      finish pytest_exception_interact --> [] [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/test_segmentation.py' lenresult=0 outcome='failed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Module test_tracker.py> [collection]
      pytest_collectstart [hook]
          collector: <Module test_tracker.py>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Module test_tracker.py>
      find_module called for: test_tracker [assertion]
      matched test file '/home/user/Documents/repos/openptv/py_bind/test/test_tracker.py' [assertion]
      found cached rewritten pyc for /home/user/Documents/repos/openptv/py_bind/test/test_tracker.py [assertion]
      early skip of rewriting module: optv.tracker [assertion]
      early skip of rewriting module: optv.parameters [assertion]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/test_tracker.py' lenresult=0 outcome='failed'> [hook]
      pytest_exception_interact [hook]
          node: <Module test_tracker.py>
          call: <CallInfo when='collect' excinfo=<ExceptionInfo CollectError("ImportError while importing test module '/home/user/Documents/repos/openptv/py_bind/test/test_tracker.py...r/Documents/repos/openptv/py_bind/optv/parameters.cpython-311-x86_64-linux-gnu.so: undefined symbol: c_read_track_par") tblen=7>>
          report: <CollectReport 'py_bind/test/test_tracker.py' lenresult=0 outcome='failed'>
      finish pytest_exception_interact --> [] [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/test_tracker.py' lenresult=0 outcome='failed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Module test_trafo_bindings.py> [collection]
      pytest_collectstart [hook]
          collector: <Module test_trafo_bindings.py>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Module test_trafo_bindings.py>
      find_module called for: test_trafo_bindings [assertion]
      matched test file '/home/user/Documents/repos/openptv/py_bind/test/test_trafo_bindings.py' [assertion]
      found cached rewritten pyc for /home/user/Documents/repos/openptv/py_bind/test/test_trafo_bindings.py [assertion]
      early skip of rewriting module: optv.parameters [assertion]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/test_trafo_bindings.py' lenresult=0 outcome='failed'> [hook]
      pytest_exception_interact [hook]
          node: <Module test_trafo_bindings.py>
          call: <CallInfo when='collect' excinfo=<ExceptionInfo CollectError("ImportError while importing test module '/home/user/Documents/repos/openptv/py_bind/test/test_trafo_bind...r/Documents/repos/openptv/py_bind/optv/parameters.cpython-311-x86_64-linux-gnu.so: undefined symbol: c_read_track_par") tblen=7>>
          report: <CollectReport 'py_bind/test/test_trafo_bindings.py' lenresult=0 outcome='failed'>
      finish pytest_exception_interact --> [] [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/test_trafo_bindings.py' lenresult=0 outcome='failed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Dir testing_fodder> [collection]
      pytest_collectstart [hook]
          collector: <Dir testing_fodder>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Dir testing_fodder>
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_directory [hook]
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers
            parent: <Dir testing_fodder>
        finish pytest_collect_directory --> <Dir burgers> [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_directory [hook]
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration
            parent: <Dir testing_fodder>
        finish pytest_collect_directory --> <Dir calibration> [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/control_parameters
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/control_parameters
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_directory [hook]
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/control_parameters
            parent: <Dir testing_fodder>
        finish pytest_collect_directory --> <Dir control_parameters> [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/corresp
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/corresp
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_directory [hook]
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/corresp
            parent: <Dir testing_fodder>
        finish pytest_collect_directory --> <Dir corresp> [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/dumbbell
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/dumbbell
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_directory [hook]
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/dumbbell
            parent: <Dir testing_fodder>
        finish pytest_collect_directory --> <Dir dumbbell> [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_directory [hook]
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame
            parent: <Dir testing_fodder>
        finish pytest_collect_directory --> <Dir frame> [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/sequence_parameters
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/sequence_parameters
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_directory [hook]
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/sequence_parameters
            parent: <Dir testing_fodder>
        finish pytest_collect_directory --> <Dir sequence_parameters> [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_directory [hook]
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam
            parent: <Dir testing_fodder>
        finish pytest_collect_directory --> <Dir single_cam> [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/target_parameters
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/target_parameters
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_directory [hook]
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/target_parameters
            parent: <Dir testing_fodder>
        finish pytest_collect_directory --> <Dir target_parameters> [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_directory [hook]
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track
            parent: <Dir testing_fodder>
        finish pytest_collect_directory --> <Dir track> [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/tracking_parameters
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/tracking_parameters
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_directory [hook]
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/tracking_parameters
            parent: <Dir testing_fodder>
        finish pytest_collect_directory --> <Dir tracking_parameters> [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/volume_parameters
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/volume_parameters
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_directory [hook]
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/volume_parameters
            parent: <Dir testing_fodder>
        finish pytest_collect_directory --> <Dir volume_parameters> [hook]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/testing_fodder' lenresult=12 outcome='passed'> [hook]
    genitems <Dir burgers> [collection]
      pytest_collectstart [hook]
          collector: <Dir burgers>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Dir burgers>
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/README.md
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/README.md
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir burgers>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/README.md
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/README.md
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/addpar.raw
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/addpar.raw
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir burgers>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/addpar.raw
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/addpar.raw
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_directory [hook]
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal
            parent: <Dir burgers>
        finish pytest_collect_directory --> <Dir cal> [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/conf.yaml
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/conf.yaml
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir burgers>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/conf.yaml
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/conf.yaml
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_directory [hook]
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig
            parent: <Dir burgers>
        finish pytest_collect_directory --> <Dir img_orig> [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/man_ori.dat
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/man_ori.dat
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir burgers>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/man_ori.dat
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/man_ori.dat
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_directory [hook]
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters
            parent: <Dir burgers>
        finish pytest_collect_directory --> <Dir parameters> [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/res_orig
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/res_orig
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_directory [hook]
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/res_orig
            parent: <Dir burgers>
        finish pytest_collect_directory --> <Dir res_orig> [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/tmp.addpar
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/tmp.addpar
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir burgers>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/tmp.addpar
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/tmp.addpar
        finish pytest_collect_file --> [] [hook]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/testing_fodder/burgers' lenresult=4 outcome='passed'> [hook]
    genitems <Dir cal> [collection]
      pytest_collectstart [hook]
          collector: <Dir cal>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Dir cal>
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam1.tif
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam1.tif
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir cal>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam1.tif
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam1.tif
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam1.tif.addpar
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam1.tif.addpar
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir cal>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam1.tif.addpar
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam1.tif.addpar
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam1.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam1.tif.ori
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir cal>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam1.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam1.tif.ori
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam2.tif
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam2.tif
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir cal>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam2.tif
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam2.tif
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam2.tif.addpar
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam2.tif.addpar
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir cal>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam2.tif.addpar
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam2.tif.addpar
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam2.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam2.tif.ori
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir cal>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam2.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam2.tif.ori
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam3.tif
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam3.tif
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir cal>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam3.tif
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam3.tif
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam3.tif.addpar
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam3.tif.addpar
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir cal>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam3.tif.addpar
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam3.tif.addpar
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam3.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam3.tif.ori
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir cal>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam3.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam3.tif.ori
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam4.tif
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam4.tif
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir cal>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam4.tif
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam4.tif
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam4.tif.addpar
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam4.tif.addpar
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir cal>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam4.tif.addpar
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam4.tif.addpar
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam4.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam4.tif.ori
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir cal>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam4.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/cam4.tif.ori
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/target_file.txt
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/target_file.txt
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir cal>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/target_file.txt
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/cal/target_file.txt
        finish pytest_collect_file --> [] [hook]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/testing_fodder/burgers/cal' lenresult=0 outcome='passed'> [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/testing_fodder/burgers/cal' lenresult=0 outcome='passed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Dir img_orig> [collection]
      pytest_collectstart [hook]
          collector: <Dir img_orig>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Dir img_orig>
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam1.10001_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam1.10001_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir img_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam1.10001_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam1.10001_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam1.10002_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam1.10002_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir img_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam1.10002_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam1.10002_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam1.10003_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam1.10003_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir img_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam1.10003_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam1.10003_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam1.10004_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam1.10004_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir img_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam1.10004_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam1.10004_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam1.10005_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam1.10005_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir img_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam1.10005_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam1.10005_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam2.10001_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam2.10001_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir img_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam2.10001_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam2.10001_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam2.10002_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam2.10002_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir img_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam2.10002_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam2.10002_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam2.10003_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam2.10003_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir img_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam2.10003_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam2.10003_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam2.10004_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam2.10004_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir img_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam2.10004_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam2.10004_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam2.10005_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam2.10005_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir img_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam2.10005_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam2.10005_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam3.10001_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam3.10001_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir img_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam3.10001_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam3.10001_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam3.10002_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam3.10002_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir img_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam3.10002_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam3.10002_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam3.10003_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam3.10003_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir img_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam3.10003_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam3.10003_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam3.10004_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam3.10004_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir img_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam3.10004_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam3.10004_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam3.10005_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam3.10005_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir img_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam3.10005_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam3.10005_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam4.10001_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam4.10001_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir img_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam4.10001_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam4.10001_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam4.10002_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam4.10002_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir img_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam4.10002_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam4.10002_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam4.10003_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam4.10003_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir img_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam4.10003_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam4.10003_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam4.10004_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam4.10004_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir img_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam4.10004_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam4.10004_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam4.10005_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam4.10005_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir img_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam4.10005_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/img_orig/cam4.10005_targets
        finish pytest_collect_file --> [] [hook]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/testing_fodder/burgers/img_orig' lenresult=0 outcome='passed'> [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/testing_fodder/burgers/img_orig' lenresult=0 outcome='passed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Dir parameters> [collection]
      pytest_collectstart [hook]
          collector: <Dir parameters>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Dir parameters>
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/cal_ori.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/cal_ori.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/cal_ori.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/cal_ori.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/criteria.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/criteria.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/criteria.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/criteria.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/detect_plate.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/detect_plate.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/detect_plate.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/detect_plate.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/dumbbell.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/dumbbell.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/dumbbell.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/dumbbell.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/examine.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/examine.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/examine.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/examine.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/man_ori.dat
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/man_ori.dat
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/man_ori.dat
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/man_ori.dat
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/man_ori.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/man_ori.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/man_ori.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/man_ori.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/multi_planes.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/multi_planes.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/multi_planes.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/multi_planes.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/orient.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/orient.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/orient.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/orient.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/pft_version
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/pft_version
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/pft_version
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/pft_version
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/pft_version.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/pft_version.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/pft_version.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/pft_version.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/ptv.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/ptv.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/ptv.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/ptv.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/sequence.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/sequence.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/sequence.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/sequence.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/shaking.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/shaking.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/shaking.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/shaking.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/sortgrid.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/sortgrid.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/sortgrid.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/sortgrid.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/targ_rec.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/targ_rec.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/targ_rec.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/targ_rec.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/track.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/track.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/track.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/track.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/unsharp_mask.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/unsharp_mask.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/unsharp_mask.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/parameters/unsharp_mask.par
        finish pytest_collect_file --> [] [hook]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/testing_fodder/burgers/parameters' lenresult=0 outcome='passed'> [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/testing_fodder/burgers/parameters' lenresult=0 outcome='passed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Dir res_orig> [collection]
      pytest_collectstart [hook]
          collector: <Dir res_orig>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Dir res_orig>
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/res_orig/rt_is.10001
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/res_orig/rt_is.10001
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir res_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/res_orig/rt_is.10001
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/res_orig/rt_is.10001
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/res_orig/rt_is.10002
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/res_orig/rt_is.10002
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir res_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/res_orig/rt_is.10002
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/res_orig/rt_is.10002
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/res_orig/rt_is.10003
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/res_orig/rt_is.10003
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir res_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/res_orig/rt_is.10003
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/res_orig/rt_is.10003
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/res_orig/rt_is.10004
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/res_orig/rt_is.10004
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir res_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/res_orig/rt_is.10004
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/res_orig/rt_is.10004
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/res_orig/rt_is.10005
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/res_orig/rt_is.10005
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir res_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/res_orig/rt_is.10005
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/res_orig/rt_is.10005
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/res_orig/rt_is.10006
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/res_orig/rt_is.10006
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir res_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/res_orig/rt_is.10006
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/burgers/res_orig/rt_is.10006
        finish pytest_collect_file --> [] [hook]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/testing_fodder/burgers/res_orig' lenresult=0 outcome='passed'> [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/testing_fodder/burgers/res_orig' lenresult=0 outcome='passed'>
      finish pytest_collectreport --> [] [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/testing_fodder/burgers' lenresult=4 outcome='passed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Dir calibration> [collection]
      pytest_collectstart [hook]
          collector: <Dir calibration>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Dir calibration>
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration/cam1.tif.addpar
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration/cam1.tif.addpar
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir calibration>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration/cam1.tif.addpar
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration/cam1.tif.addpar
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration/cam1.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration/cam1.tif.ori
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir calibration>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration/cam1.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration/cam1.tif.ori
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration/cam2.tif.addpar
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration/cam2.tif.addpar
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir calibration>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration/cam2.tif.addpar
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration/cam2.tif.addpar
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration/sym_cam1.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration/sym_cam1.tif.ori
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir calibration>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration/sym_cam1.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration/sym_cam1.tif.ori
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration/sym_cam2.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration/sym_cam2.tif.ori
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir calibration>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration/sym_cam2.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration/sym_cam2.tif.ori
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration/sym_cam3.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration/sym_cam3.tif.ori
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir calibration>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration/sym_cam3.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration/sym_cam3.tif.ori
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration/sym_cam4.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration/sym_cam4.tif.ori
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir calibration>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration/sym_cam4.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/calibration/sym_cam4.tif.ori
        finish pytest_collect_file --> [] [hook]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/testing_fodder/calibration' lenresult=0 outcome='passed'> [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/testing_fodder/calibration' lenresult=0 outcome='passed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Dir control_parameters> [collection]
      pytest_collectstart [hook]
          collector: <Dir control_parameters>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Dir control_parameters>
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/control_parameters/control.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/control_parameters/control.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir control_parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/control_parameters/control.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/control_parameters/control.par
        finish pytest_collect_file --> [] [hook]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/testing_fodder/control_parameters' lenresult=0 outcome='passed'> [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/testing_fodder/control_parameters' lenresult=0 outcome='passed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Dir corresp> [collection]
      pytest_collectstart [hook]
          collector: <Dir corresp>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Dir corresp>
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/corresp/control.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/corresp/control.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir corresp>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/corresp/control.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/corresp/control.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/corresp/criteria.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/corresp/criteria.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir corresp>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/corresp/criteria.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/corresp/criteria.par
        finish pytest_collect_file --> [] [hook]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/testing_fodder/corresp' lenresult=0 outcome='passed'> [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/testing_fodder/corresp' lenresult=0 outcome='passed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Dir dumbbell> [collection]
      pytest_collectstart [hook]
          collector: <Dir dumbbell>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Dir dumbbell>
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/dumbbell/cam1.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/dumbbell/cam1.tif.ori
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir dumbbell>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/dumbbell/cam1.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/dumbbell/cam1.tif.ori
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/dumbbell/cam2.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/dumbbell/cam2.tif.ori
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir dumbbell>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/dumbbell/cam2.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/dumbbell/cam2.tif.ori
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/dumbbell/cam3.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/dumbbell/cam3.tif.ori
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir dumbbell>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/dumbbell/cam3.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/dumbbell/cam3.tif.ori
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/dumbbell/cam4.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/dumbbell/cam4.tif.ori
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir dumbbell>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/dumbbell/cam4.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/dumbbell/cam4.tif.ori
        finish pytest_collect_file --> [] [hook]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/testing_fodder/dumbbell' lenresult=0 outcome='passed'> [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/testing_fodder/dumbbell' lenresult=0 outcome='passed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Dir frame> [collection]
      pytest_collectstart [hook]
          collector: <Dir frame>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Dir frame>
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/added.333
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/added.333
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir frame>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/added.333
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/added.333
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/cam1.0333_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/cam1.0333_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir frame>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/cam1.0333_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/cam1.0333_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/cam1_reversed.0333_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/cam1_reversed.0333_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir frame>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/cam1_reversed.0333_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/cam1_reversed.0333_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/cam2.0333_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/cam2.0333_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir frame>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/cam2.0333_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/cam2.0333_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/cam3.0333_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/cam3.0333_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir frame>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/cam3.0333_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/cam3.0333_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/cam4.0333_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/cam4.0333_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir frame>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/cam4.0333_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/cam4.0333_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/ptv_is.333
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/ptv_is.333
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir frame>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/ptv_is.333
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/ptv_is.333
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/rt_is.333
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/rt_is.333
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir frame>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/rt_is.333
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/frame/rt_is.333
        finish pytest_collect_file --> [] [hook]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/testing_fodder/frame' lenresult=0 outcome='passed'> [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/testing_fodder/frame' lenresult=0 outcome='passed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Dir sequence_parameters> [collection]
      pytest_collectstart [hook]
          collector: <Dir sequence_parameters>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Dir sequence_parameters>
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/sequence_parameters/sequence.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/sequence_parameters/sequence.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir sequence_parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/sequence_parameters/sequence.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/sequence_parameters/sequence.par
        finish pytest_collect_file --> [] [hook]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/testing_fodder/sequence_parameters' lenresult=0 outcome='passed'> [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/testing_fodder/sequence_parameters' lenresult=0 outcome='passed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Dir single_cam> [collection]
      pytest_collectstart [hook]
          collector: <Dir single_cam>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Dir single_cam>
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/calibration
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/calibration
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_directory [hook]
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/calibration
            parent: <Dir single_cam>
        finish pytest_collect_directory --> <Dir calibration> [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_directory [hook]
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img
            parent: <Dir single_cam>
        finish pytest_collect_directory --> <Dir img> [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_directory [hook]
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters
            parent: <Dir single_cam>
        finish pytest_collect_directory --> <Dir parameters> [hook]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/testing_fodder/single_cam' lenresult=3 outcome='passed'> [hook]
    genitems <Dir calibration> [collection]
      pytest_collectstart [hook]
          collector: <Dir calibration>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Dir calibration>
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/calibration/calblock.txt
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/calibration/calblock.txt
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir calibration>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/calibration/calblock.txt
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/calibration/calblock.txt
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/calibration/cam_1.tif.addpar
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/calibration/cam_1.tif.addpar
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir calibration>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/calibration/cam_1.tif.addpar
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/calibration/cam_1.tif.addpar
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/calibration/cam_1.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/calibration/cam_1.tif.ori
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir calibration>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/calibration/cam_1.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/calibration/cam_1.tif.ori
        finish pytest_collect_file --> [] [hook]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/testing_fodder/single_cam/calibration' lenresult=0 outcome='passed'> [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/testing_fodder/single_cam/calibration' lenresult=0 outcome='passed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Dir img> [collection]
      pytest_collectstart [hook]
          collector: <Dir img>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Dir img>
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img/img.10004
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img/img.10004
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir img>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img/img.10004
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img/img.10004
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img/img.10005
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img/img.10005
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir img>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img/img.10005
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img/img.10005
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img/img.10006
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img/img.10006
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir img>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img/img.10006
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img/img.10006
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img/img.10007
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img/img.10007
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir img>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img/img.10007
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img/img.10007
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img/img.10008
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img/img.10008
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir img>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img/img.10008
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img/img.10008
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img/img.10009
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img/img.10009
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir img>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img/img.10009
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img/img.10009
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img/img.10010
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img/img.10010
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir img>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img/img.10010
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/img/img.10010
        finish pytest_collect_file --> [] [hook]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/testing_fodder/single_cam/img' lenresult=0 outcome='passed'> [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/testing_fodder/single_cam/img' lenresult=0 outcome='passed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Dir parameters> [collection]
      pytest_collectstart [hook]
          collector: <Dir parameters>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Dir parameters>
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/cal_ori.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/cal_ori.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/cal_ori.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/cal_ori.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/criteria.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/criteria.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/criteria.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/criteria.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/detect_plate.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/detect_plate.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/detect_plate.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/detect_plate.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/dumbbell.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/dumbbell.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/dumbbell.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/dumbbell.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/examine.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/examine.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/examine.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/examine.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/man_ori.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/man_ori.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/man_ori.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/man_ori.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/multi_planes.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/multi_planes.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/multi_planes.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/multi_planes.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/orient.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/orient.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/orient.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/orient.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/pft_version.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/pft_version.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/pft_version.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/pft_version.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/ptv.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/ptv.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/ptv.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/ptv.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/sequence.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/sequence.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/sequence.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/sequence.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/shaking.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/shaking.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/shaking.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/shaking.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/sortgrid.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/sortgrid.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/sortgrid.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/sortgrid.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/targ_rec.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/targ_rec.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/targ_rec.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/targ_rec.par
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/track.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/track.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/track.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/single_cam/parameters/track.par
        finish pytest_collect_file --> [] [hook]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/testing_fodder/single_cam/parameters' lenresult=0 outcome='passed'> [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/testing_fodder/single_cam/parameters' lenresult=0 outcome='passed'>
      finish pytest_collectreport --> [] [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/testing_fodder/single_cam' lenresult=3 outcome='passed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Dir target_parameters> [collection]
      pytest_collectstart [hook]
          collector: <Dir target_parameters>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Dir target_parameters>
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/target_parameters/targ_rec.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/target_parameters/targ_rec.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir target_parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/target_parameters/targ_rec.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/target_parameters/targ_rec.par
        finish pytest_collect_file --> [] [hook]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/testing_fodder/target_parameters' lenresult=0 outcome='passed'> [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/testing_fodder/target_parameters' lenresult=0 outcome='passed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Dir track> [collection]
      pytest_collectstart [hook]
          collector: <Dir track>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Dir track>
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/cal
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/cal
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_directory [hook]
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/cal
            parent: <Dir track>
        finish pytest_collect_directory --> <Dir cal> [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/conf.yaml
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/conf.yaml
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir track>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/conf.yaml
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/conf.yaml
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_directory [hook]
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart
            parent: <Dir track>
        finish pytest_collect_directory --> <Dir newpart> [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/res_orig
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/res_orig
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_directory [hook]
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/res_orig
            parent: <Dir track>
        finish pytest_collect_directory --> <Dir res_orig> [hook]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/testing_fodder/track' lenresult=3 outcome='passed'> [hook]
    genitems <Dir cal> [collection]
      pytest_collectstart [hook]
          collector: <Dir cal>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Dir cal>
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/cal/cam1.tif.addpar
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/cal/cam1.tif.addpar
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir cal>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/cal/cam1.tif.addpar
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/cal/cam1.tif.addpar
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/cal/sym_cam1.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/cal/sym_cam1.tif.ori
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir cal>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/cal/sym_cam1.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/cal/sym_cam1.tif.ori
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/cal/sym_cam2.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/cal/sym_cam2.tif.ori
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir cal>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/cal/sym_cam2.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/cal/sym_cam2.tif.ori
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/cal/sym_cam3.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/cal/sym_cam3.tif.ori
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir cal>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/cal/sym_cam3.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/cal/sym_cam3.tif.ori
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/cal/sym_cam4.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/cal/sym_cam4.tif.ori
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir cal>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/cal/sym_cam4.tif.ori
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/cal/sym_cam4.tif.ori
        finish pytest_collect_file --> [] [hook]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/testing_fodder/track/cal' lenresult=0 outcome='passed'> [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/testing_fodder/track/cal' lenresult=0 outcome='passed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Dir newpart> [collection]
      pytest_collectstart [hook]
          collector: <Dir newpart>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Dir newpart>
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam1.10000_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam1.10000_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir newpart>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam1.10000_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam1.10000_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam1.10001_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam1.10001_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir newpart>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam1.10001_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam1.10001_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam1.10002_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam1.10002_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir newpart>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam1.10002_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam1.10002_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam1.10003_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam1.10003_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir newpart>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam1.10003_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam1.10003_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam1.10004_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam1.10004_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir newpart>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam1.10004_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam1.10004_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam1.10005_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam1.10005_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir newpart>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam1.10005_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam1.10005_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam2.10000_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam2.10000_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir newpart>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam2.10000_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam2.10000_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam2.10001_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam2.10001_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir newpart>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam2.10001_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam2.10001_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam2.10002_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam2.10002_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir newpart>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam2.10002_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam2.10002_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam2.10003_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam2.10003_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir newpart>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam2.10003_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam2.10003_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam2.10004_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam2.10004_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir newpart>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam2.10004_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam2.10004_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam2.10005_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam2.10005_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir newpart>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam2.10005_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam2.10005_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam3.10000_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam3.10000_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir newpart>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam3.10000_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam3.10000_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam3.10001_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam3.10001_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir newpart>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam3.10001_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam3.10001_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam3.10002_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam3.10002_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir newpart>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam3.10002_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam3.10002_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam3.10003_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam3.10003_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir newpart>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam3.10003_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam3.10003_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam3.10004_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam3.10004_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir newpart>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam3.10004_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam3.10004_targets
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam3.10005_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam3.10005_targets
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir newpart>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam3.10005_targets
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/newpart/cam3.10005_targets
        finish pytest_collect_file --> [] [hook]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/testing_fodder/track/newpart' lenresult=0 outcome='passed'> [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/testing_fodder/track/newpart' lenresult=0 outcome='passed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Dir res_orig> [collection]
      pytest_collectstart [hook]
          collector: <Dir res_orig>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Dir res_orig>
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/res_orig/particles.10001
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/res_orig/particles.10001
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir res_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/res_orig/particles.10001
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/res_orig/particles.10001
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/res_orig/particles.10002
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/res_orig/particles.10002
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir res_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/res_orig/particles.10002
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/res_orig/particles.10002
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/res_orig/particles.10003
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/res_orig/particles.10003
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir res_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/res_orig/particles.10003
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/res_orig/particles.10003
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/res_orig/particles.10004
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/res_orig/particles.10004
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir res_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/res_orig/particles.10004
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/res_orig/particles.10004
        finish pytest_collect_file --> [] [hook]
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/res_orig/particles.10005
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/res_orig/particles.10005
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir res_orig>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/res_orig/particles.10005
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/track/res_orig/particles.10005
        finish pytest_collect_file --> [] [hook]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/testing_fodder/track/res_orig' lenresult=0 outcome='passed'> [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/testing_fodder/track/res_orig' lenresult=0 outcome='passed'>
      finish pytest_collectreport --> [] [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/testing_fodder/track' lenresult=3 outcome='passed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Dir tracking_parameters> [collection]
      pytest_collectstart [hook]
          collector: <Dir tracking_parameters>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Dir tracking_parameters>
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/tracking_parameters/track.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/tracking_parameters/track.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir tracking_parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/tracking_parameters/track.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/tracking_parameters/track.par
        finish pytest_collect_file --> [] [hook]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/testing_fodder/tracking_parameters' lenresult=0 outcome='passed'> [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/testing_fodder/tracking_parameters' lenresult=0 outcome='passed'>
      finish pytest_collectreport --> [] [hook]
    genitems <Dir volume_parameters> [collection]
      pytest_collectstart [hook]
          collector: <Dir volume_parameters>
      finish pytest_collectstart --> [] [hook]
      pytest_make_collect_report [hook]
          collector: <Dir volume_parameters>
        pytest_ignore_collect [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            collection_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/volume_parameters/volume.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/volume_parameters/volume.par
        finish pytest_ignore_collect --> None [hook]
        pytest_collect_file [hook]
            parent: <Dir volume_parameters>
            file_path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/volume_parameters/volume.par
            path: /home/user/Documents/repos/openptv/py_bind/test/testing_fodder/volume_parameters/volume.par
        finish pytest_collect_file --> [] [hook]
      finish pytest_make_collect_report --> <CollectReport 'py_bind/test/testing_fodder/volume_parameters' lenresult=0 outcome='passed'> [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/testing_fodder/volume_parameters' lenresult=0 outcome='passed'>
      finish pytest_collectreport --> [] [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test/testing_fodder' lenresult=12 outcome='passed'>
      finish pytest_collectreport --> [] [hook]
      pytest_collectreport [hook]
          report: <CollectReport 'py_bind/test' lenresult=13 outcome='passed'>
      finish pytest_collectreport --> [] [hook]
      pytest_collection_modifyitems [hook]
          session: <Session  exitstatus=<ExitCode.OK: 0> testsfailed=10 testscollected=0>
          config: <_pytest.config.Config object at 0x7fa936766950>
          items: [<TestCaseFunction test_Calibration_instantiation>, <TestCaseFunction test_full_instantiate>, <TestCaseFunction test_set_angles>, <TestCaseFunction test_set_decentering>, <TestCaseFunction test_set_glass>, <TestCaseFunction test_set_pos>, <TestCaseFunction test_set_primary>, <TestCaseFunction test_set_radial>, <TestCaseFunction test_fill_target>, <TestCaseFunction test_fill_target_array>, <TestCaseFunction test_read_targets>, <TestCaseFunction test_sort_y>, <TestCaseFunction test_write_targets>, <TestCaseFunction test_read_frame>]
      finish pytest_collection_modifyitems --> [] [hook]
      pytest_collection_finish [hook]
          session: <Session  exitstatus=<ExitCode.OK: 0> testsfailed=10 testscollected=0>
        pytest_report_collectionfinish [hook]
            config: <_pytest.config.Config object at 0x7fa936766950>
            items: [<TestCaseFunction test_Calibration_instantiation>, <TestCaseFunction test_full_instantiate>, <TestCaseFunction test_set_angles>, <TestCaseFunction test_set_decentering>, <TestCaseFunction test_set_glass>, <TestCaseFunction test_set_pos>, <TestCaseFunction test_set_primary>, <TestCaseFunction test_set_radial>, <TestCaseFunction test_fill_target>, <TestCaseFunction test_fill_target_array>, <TestCaseFunction test_read_targets>, <TestCaseFunction test_sort_y>, <TestCaseFunction test_write_targets>, <TestCaseFunction test_read_frame>]
            start_path: /home/user/Documents/repos/openptv/py_bind/test
            startdir: /home/user/Documents/repos/openptv/py_bind/test
        finish pytest_report_collectionfinish --> [] [hook]
      finish pytest_collection_finish --> [] [hook]
    finish pytest_collection --> None [hook]
    pytest_runtestloop [hook]
        session: <Session  exitstatus=<ExitCode.OK: 0> testsfailed=10 testscollected=14>
    pytest_keyboard_interrupt [hook]
        excinfo: <ExceptionInfo Interrupted('10 errors during collection') tblen=15>
    finish pytest_keyboard_interrupt --> [] [hook]
    pytest_sessionfinish [hook]
        session: <Session  exitstatus=<ExitCode.INTERRUPTED: 2> testsfailed=10 testscollected=14>
        exitstatus: 2
      pytest_terminal_summary [hook]
          terminalreporter: <_pytest.terminal.TerminalReporter object at 0x7fa935d23b90>
          exitstatus: 2
          config: <_pytest.config.Config object at 0x7fa936766950>
        pytest_report_teststatus [hook]
            report: <CollectReport 'py_bind/test/test_burgers.py' lenresult=0 outcome='failed'>
            config: <_pytest.config.Config object at 0x7fa936766950>
        finish pytest_report_teststatus --> ('error', 'E', 'ERROR') [hook]
        pytest_report_teststatus [hook]
            report: <CollectReport 'py_bind/test/test_corresp.py' lenresult=0 outcome='failed'>
            config: <_pytest.config.Config object at 0x7fa936766950>
        finish pytest_report_teststatus --> ('error', 'E', 'ERROR') [hook]
        pytest_report_teststatus [hook]
            report: <CollectReport 'py_bind/test/test_epipolar.py' lenresult=0 outcome='failed'>
            config: <_pytest.config.Config object at 0x7fa936766950>
        finish pytest_report_teststatus --> ('error', 'E', 'ERROR') [hook]
        pytest_report_teststatus [hook]
            report: <CollectReport 'py_bind/test/test_image_processing.py' lenresult=0 outcome='failed'>
            config: <_pytest.config.Config object at 0x7fa936766950>
        finish pytest_report_teststatus --> ('error', 'E', 'ERROR') [hook]
        pytest_report_teststatus [hook]
            report: <CollectReport 'py_bind/test/test_img_coord.py' lenresult=0 outcome='failed'>
            config: <_pytest.config.Config object at 0x7fa936766950>
        finish pytest_report_teststatus --> ('error', 'E', 'ERROR') [hook]
        pytest_report_teststatus [hook]
            report: <CollectReport 'py_bind/test/test_orientation.py' lenresult=0 outcome='failed'>
            config: <_pytest.config.Config object at 0x7fa936766950>
        finish pytest_report_teststatus --> ('error', 'E', 'ERROR') [hook]
        pytest_report_teststatus [hook]
            report: <CollectReport 'py_bind/test/test_parameters_bindings.py' lenresult=0 outcome='failed'>
            config: <_pytest.config.Config object at 0x7fa936766950>
        finish pytest_report_teststatus --> ('error', 'E', 'ERROR') [hook]
        pytest_report_teststatus [hook]
            report: <CollectReport 'py_bind/test/test_segmentation.py' lenresult=0 outcome='failed'>
            config: <_pytest.config.Config object at 0x7fa936766950>
        finish pytest_report_teststatus --> ('error', 'E', 'ERROR') [hook]
        pytest_report_teststatus [hook]
            report: <CollectReport 'py_bind/test/test_tracker.py' lenresult=0 outcome='failed'>
            config: <_pytest.config.Config object at 0x7fa936766950>
        finish pytest_report_teststatus --> ('error', 'E', 'ERROR') [hook]
        pytest_report_teststatus [hook]
            report: <CollectReport 'py_bind/test/test_trafo_bindings.py' lenresult=0 outcome='failed'>
            config: <_pytest.config.Config object at 0x7fa936766950>
        finish pytest_report_teststatus --> ('error', 'E', 'ERROR') [hook]
      finish pytest_terminal_summary --> [] [hook]
    finish pytest_sessionfinish --> [] [hook]
    pytest_unconfigure [hook]
        config: <_pytest.config.Config object at 0x7fa936766950>
    finish pytest_unconfigure --> [] [hook]
