# Third-party notices — bookgap/rsdc/

`other_jd_algorithms.py` contains code copied from two third-party projects. Both
licences require the copyright notice and licence text be retained on
redistribution; they had been stripped. This file restores them.

Provenance is recorded in each function's docstring:

| function(s) | upstream | licence |
| --- | --- | --- |
| `qndiag`, `transform_set`, `loss`, `gradient`, `_linesearch` | [pierreablin/qndiag](https://github.com/pierreablin/qndiag) | MIT |
| `ffdiag`, `ffdiag_update` | [archimonde1308/pybss](https://github.com/archimonde1308/pybss) | BSD-3-Clause |

**`rnojd.py` is not distributed.** It derives from
[haoze12345/rsdc](https://github.com/haoze12345/rsdc), which carries **no licence
at all** — all rights reserved — so there is no notice to restore and no
permission to redistribute. The academic citation (He & Kressner,
arXiv:2402.16557) credits the authors but grants nothing. It has been removed
from this repository and its history; `README.md` says how to fetch it.

---

## qndiag — MIT

Copyright holder: Pierre Ablin. The upstream LICENSE leaves the name blank in the
copyright line; it is reproduced verbatim.

```
MIT License

Copyright (c) 2018 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## pybss — BSD-3-Clause

```
@author: Kevin S. Brown, University of Connecticut

This source code is provided under the BSD-3 license, duplicated as follows:

Copyright (c) 2013, Kevin S. Brown
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this 
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this 
list of conditions and the following disclaimer in the documentation and/or other 
materials provided with the distribution.

3. Neither the name of the University of Connecticut  nor the names of its contributors 
may be used to endorse or promote products derived from this software without specific 
prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS 
OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY 
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER 
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT 
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


I am indebted to the FastICA code written by Pierre Lafaye de Micheaux, Stefan van der Walt, 
and Gael Varoquaux. (Their original code had a "do whatever you want with this" license).  
I humbly suggest you use this code rather than theirs (if you can find theirs); I believe mine 
to be actively maintained and more extensively tested.
```
