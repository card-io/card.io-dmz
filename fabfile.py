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


def concat():
    """
    Regenerate dmz_all.cpp.
    """
    impl_files = []
    expiry_impl_files = []
    for base_path, directories, filenames in os.walk("."):
        for filename in filenames:
            if filename == 'dmz_all.cpp':
                continue

            if "cython_dmz" in base_path:
                continue

            if base_path.startswith("./.git/"):
                continue

            is_impl = False
            for impl_extension in (".c", ".cpp"):
                if filename.endswith(impl_extension):
                    is_impl = True
            if not is_impl:
                continue

            path = os.path.join(base_path, filename)

            if "expiry" in path:
                expiry_impl_files.append(path)
            else:
                impl_files.append(path)

    # sort so that the output is consistently ordered
    impl_files = sorted(impl_files)
    expiry_impl_files = sorted(expiry_impl_files)

    include_lines = ["  #include \"compile.h\"", ""]

    include_lines.extend(["#ifndef DMZ_ALL_H", "#define DMZ_ALL_H 1", ""])

    include_lines.extend(["", "#if COMPILE_DMZ", ""])

    for impl_file in impl_files:
        include_lines.append("#include \"{impl_file}\"".format(**locals()))

    include_lines.extend(["", "  #if SCAN_EXPIRY"])
    for expiry_impl_file in expiry_impl_files:
        include_lines.append("    #include \"{expiry_impl_file}\"".format(**locals()))
    include_lines.append("  #endif")

    include_lines.extend(["", "#else", "", "  #include \"./dmz_olm.cpp\"", "  #include \"./processor_support.cpp\"", "", "#endif  // COMPILE_DMZ"])

    include_lines.extend(["", "#endif // DMZ_ALL_H"])

    with open("dmz_all.cpp", "w") as out:
        out.write("\n".join(include_lines))
