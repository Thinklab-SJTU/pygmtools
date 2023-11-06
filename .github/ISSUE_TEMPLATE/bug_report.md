---
name: Bug report
about: Create a report to help us improve
title: "[BUG]"
labels: ''
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Code to reproduce the behavior:
```
import pygmtools as pygm
import numpy as np
m = np.zeros(100, 100)
pygm.some_error_func(m) # error!
```

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment:**
 - OS: [e.g. Ubuntu 22.04]
 - Pygmtools Version [e.g. 0.4.0]
 - Environment report, by running the command ``python3 -c 'import pygmtools; pygmtools.env_report()'``

**Additional context**
Add any other context about the problem here.
