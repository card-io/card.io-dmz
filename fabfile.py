#!/usr/bin/env python

import hashlib
import os
import re

from fabric.api import env, local, hide
from fabric.context_managers import lcd, settings
from fabric.contrib.console import confirm
from fabric.contrib.files import exists
from fabric.decorators import runs_once
from fabric.utils import abort
from fabric import colors


SALT = "i wish i were a hunter in search of different food"

env.verbose = False

def verbose(be_verbose=True):
    """
    Makes all following tasks more verbose.
    """
    env.verbose = be_verbose


def concat():
    """
    Regenerate dmz_all.cpp.
    """
    impl_files = []
    for base_path, directories, filenames in os.walk("."):
        for filename in filenames:
            if filename == 'dmz_all.cpp':
                continue
                
            if base_path.startswith("./.git/"):
                continue

            is_impl = False
            for impl_extension in (".c", ".cpp"):
                if filename.endswith(impl_extension):
                    is_impl = True
            if not is_impl:
                continue

            impl_files.append(os.path.join(base_path, filename))

    # sort so that the output is consistently ordered
    impl_files = sorted(impl_files)

    include_lines = ["#include \"compile.h\"", ""]

    include_lines.extend(["#ifndef DMZ_ALL_H", "#define DMZ_ALL_H 1", ""])

    include_lines.extend(["", "#if COMPILE_DMZ", ""])
                          
    for impl_file in impl_files:
        include_lines.append("#include \"{impl_file}\"".format(**locals()))

    include_lines.extend(["", "#else", "", "#include \"./dmz_olm.cpp\"", "#include \"./processor_support.cpp\"", "", "#endif  // COMPILE_DMZ"])

    include_lines.extend(["", "#endif // DMZ_ALL_H"])

    with open("dmz_all.cpp", "w") as out:
        out.write("\n".join(include_lines))

