#!/bin/bash

# ---------------------------------------------------------------------------
#  Copyright (C) 2019-2022 Intel Corporation. All rights reserved
#  ---------------------------------------------------------------------------
# Usage: source cltools_setenv.sh [ma2x8x/3010xx]
# The script takes the SoC variant as an argument

source_name=$( readlink -f ${BASH_SOURCE})
SHAVE_CL_INSTALL=${source_name%/*}/..
if ([[ $1'X' != '3010xxX' && $1'X' != 'ma2x8xX' ]]); then
    echo "Usage: source cltools_setenv.sh [ma2x8x/3010xx]";
else
    export SHAVE_LDSCRIPT_DIR="$SHAVE_CL_INSTALL/ldscripts/$1"
    export SHAVE_MA2X8XLIBS_DIR="$SHAVE_CL_INSTALL/lib"
    export SHAVE_MYRIAD_LD_DIR="$SHAVE_CL_INSTALL/bin";
fi
