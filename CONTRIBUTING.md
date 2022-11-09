# Contributing to pygmtools

First, thank you for contributing to ``pygmtools``! 

## How to contribute

The preferred workflow for contributing to ``pygmtools`` is to fork the
[main repository](https://github.com/Thinklab-SJTU/pygmtools) on
GitHub, clone, and develop on a branch. Steps:

1. Fork the [project repository](https://github.com/Thinklab-SJTU/pygmtools)
   by clicking on the 'Fork' button near the top right of the page. This creates
   a copy of the code under your GitHub user account. For more details on
   how to fork a repository see [this guide](https://help.github.com/articles/fork-a-repo/).

2. Clone your fork of the repo from your GitHub account to your local disk:

   ```bash
   $ git clone git@github.com:YourUserName/pygmtools.git
   $ cd pygmtools
   ```

3. Create a ``feature`` branch to hold your development changes:

   ```bash
   $ git checkout -b my-feature
   ```

   Always use a ``feature`` branch. It is good practice to never work on the ``master`` branch!

4. Develop the feature on your feature branch. Add changed files using ``git add`` and then ``git commit`` files:

   ```bash
   $ git add modified_files
   $ git commit
   ```

   to record your changes in Git, then push the changes to your GitHub account with:

   ```bash
   $ git push -u origin my-feature
   ```

5. Follow [these instructions](https://help.github.com/articles/creating-a-pull-request-from-a-fork)
to create a pull request from your fork. This will email the committers and an automatic check will run.

(If any of the above seems like magic to you, please look up the
[Git documentation](https://git-scm.com/documentation) on the web, or ask a friend or another contributor for help.)

## Pull Request Checklist

We recommended that your contribution complies with the
following rules before you submit a pull request:

-  Follow the PEP8 Guidelines.

-  If your pull request addresses an issue, please use the pull request title
   to describe the issue and mention the issue number in the pull request description. This will make sure a link back to the original issue is
   created.

-  All public methods should have informative docstrings with sample
   usage presented as doctests when appropriate.

-  When adding additional functionality, provide at least one
   example script in the ``examples/`` folder. Have a look at other
   examples for reference. Examples should demonstrate why the new
   functionality is useful in practice and, if possible, compare it
   to other methods available in ``pygmtools``.

-  Documentation and high-coverage tests are necessary for enhancements to be
   accepted. Bug-fixes or new features should be provided with 
   [non-regression tests](https://en.wikipedia.org/wiki/Non-regression_testing).
   These tests verify the correct behavior of the fix or feature. In this
   manner, further modifications on the code base are granted to be consistent
   with the desired behavior.
   For the Bug-fixes case, at the time of the PR, these tests should fail for
   the code base in master and pass for the PR code.

-  At least one paragraph of narrative documentation with links to
   references in the literature and
   the example.

You can also check for common programming errors with the following
tools:


-  No pyflakes warnings, check with:

  ```bash
  $ pip install pyflakes
  $ pyflakes path/to/module.py
  ```

-  No PEP8 warnings, check with:

  ```bash
  $ pip install pep8
  $ pep8 path/to/module.py
  ```

-  AutoPEP8 can help you fix some of the easy redundant errors:

  ```bash
  $ pip install autopep8
  $ autopep8 path/to/pep8.py
  ```

## Filing bugs

We use Github issues to track all bugs and feature requests; feel free to
open an issue if you have found a bug or wish to see a feature implemented.

It is recommended to check that your issue complies with the
following rules before submitting:

-  Verify that your issue is not being currently addressed by other
   [issues](https://github.com/Thinklab-SJTU/pygmtools/issues?q=)
   or [pull requests](https://github.com/Thinklab-SJTU/pygmtools/pulls?q=).

-  Please ensure all code snippets and error messages are formatted in
   appropriate code blocks.
   See [Creating and highlighting code blocks](https://help.github.com/articles/creating-and-highlighting-code-blocks).

-  Please include your operating system type and version number, as well
   as your Python, pygmtools, numpy, and scipy versions. Please also provide 
   the name of your running backend, and the GPU/CUDA versions if you are using GPU.
   This information can be found by running the following environment report (``pygmtools>=0.2.9``):
   
   ```bash
   $ python3 -c 'import pygmtools; pygmtools.env_report()'
   ```
   If you are using GPU, make sure to install ``pynvml`` before running the above 
   script: ``pip install pynvml``.

-  Please be specific about what estimators and/or functions are involved
   and the shape of the data, as appropriate; please include a
   [reproducible](http://stackoverflow.com/help/mcve) code snippet
   or link to a [gist](https://gist.github.com). If an exception is raised,
   please provide the traceback.
   

## Documentation

We are glad to accept any sort of documentation: function docstrings,
reStructuredText documents, tutorials, examples, etc.
reStructuredText documents live in the source code repository under the
``doc/`` directory.

You can edit the documentation using any text editor and then generate
the HTML output by typing ``make html`` from the ``docs/`` directory. 
The resulting HTML files are in ``docs/_build/`` and are viewable in 
any web browser. The example files in ``examples/`` are also built.
If you want to skip building the examples, please use the command
``make html-noplot``.

For building the documentation, you will need the packages listed in
``docs/requirements.txt``. Please use ``python>=3.7`` because the packages
for earlier Python versions are outdated.

When you are writing documentation, it is important to keep a good
compromise between mathematical and algorithmic details, and give
intuition to the reader on what the algorithm does. It is best to always
start with a small paragraph with a hand-waving explanation of what the
method does to the data.


This Contribution guide is strongly inpired by the one of the [scikit-learn](https://github.com/scikit-learn/scikit-learn) team.