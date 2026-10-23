// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
using Experiments;
using Renci.SshNet.Messages.Connection;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ozAIE.Experiments
{
    public class TestsConfig
    {
        public MeasurementType TestsType { get; set; }
        public bool pinBLISThreadsToCore { get; set; }
        public bool resume { get; set; } = false;
        public string ssh_config_dir { get; set; } = "";
        public string mlir_repo_dir { get; set; } = "";
        public int l_mant { get; set; } = 53;
        public int min_K { get; set; } = 7;
        public int max_K { get; set; } = 11;
        public string matlab_bin { get; set; } = "";
    }
}
