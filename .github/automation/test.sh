#! /bin/bash

#===============================================================================
# Copyright 2019-2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --build-dir)
        BUILD_DIR="$2"
        ;;
        --report-dir)
        REPORT_DIR="$2"
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
    shift
done


if [ "$(uname)" == "Linux" ]; then
    export OMP_NUM_THREADS="$(grep -c processor /proc/cpuinfo)"
else
    export OMP_NUM_THREADS="$(sysctl -n hw.physicalcpu)"
fi

if  [[ ! -z "${REPORT_DIR}" ]]; then
    export GTEST_OUTPUT="${REPORT_DIR}/report/test_report.xml"
fi

#  We currently have some OS and config specific test failures.
if [[ "$(uname)" == "Linux" ]]; then
    SKIPPED_TEST_FAILURES="test_iface_attr"
    SKIPPED_TEST_FAILURES+="|test_pooling_forward"
    SKIPPED_TEST_FAILURES+="|test_api"
    SKIPPED_TEST_FAILURES+="|test_graph_unit_dnnl_conv_usm_cpu"
    SKIPPED_TEST_FAILURES+="|test_graph_unit_dnnl_matmul_usm_cpu"
    SKIPPED_TEST_FAILURES+="|test_graph_unit_dnnl_pool_usm_cpu"
    SKIPPED_TEST_FAILURES+="|test_benchdnn_modeC_binary_smoke_cpu"
    SKIPPED_TEST_FAILURES+="|test_benchdnn_modeC_concat_smoke_cpu"
    SKIPPED_TEST_FAILURES+="|test_benchdnn_modeC_conv_smoke_cpu"
    SKIPPED_TEST_FAILURES+="|test_benchdnn_modeC_deconv_smoke_cpu"
    SKIPPED_TEST_FAILURES+="|test_benchdnn_modeC_pool_smoke_cpu"
    SKIPPED_TEST_FAILURES+="|test_benchdnn_modeC_reorder_smoke_cpu"
fi

CTEST_OPTS="${CTEST_OPTS} --verbose --output-on-failure -E $SKIPPED_TEST_FAILURES"

echo "CTest options: ${CTEST_OPTS}"
cd "${BUILD_DIR}"
ctest ${CTEST_OPTS}
result=$?

echo "DONE"
exit $result
