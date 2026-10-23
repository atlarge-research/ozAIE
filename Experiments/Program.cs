// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

using ozAIE.Experiments;
using Org.BouncyCastle.Asn1;
using Org.BouncyCastle.Crypto;
using Renci.SshNet;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Numerics;
using System.Reflection.Metadata.Ecma335;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading.Tasks.Dataflow;
using static System.Net.Mime.MediaTypeNames;
using Experiments;

int pauseBetweenTestMeasurementsMillis = 10000;

var jsonOptions = new JsonSerializerOptions
{
    WriteIndented = true
};

Dictionary<Device, string> hostsMap = new Dictionary<Device, string>();
hostsMap[Device.npu1] = "xdna";
hostsMap[Device.npu2] = "xdna2";

Dictionary<Device, int> cpuCoresMap = new Dictionary<Device, int>();
cpuCoresMap[Device.npu1] = 8;
cpuCoresMap[Device.npu2] = 16;

Dictionary<Device, int[]> cpu_affinity = new Dictionary<Device, int[]>();
cpu_affinity[Device.npu1] = Enumerable.Range(0, 8).ToArray();
cpu_affinity[Device.npu2] = Enumerable.Range(0, 16).Select(i => i*2).ToArray();

Dictionary<CoreConfig, string> directoriesMap = new Dictionary<CoreConfig, string>();
directoriesMap[CoreConfig.single] = "single_core";
directoriesMap[CoreConfig.whole] = "whole_array";

Dictionary<Func<TestsConfig, bool>, string[]> testTypeToScriptsMap = new Dictionary<Func<TestsConfig, bool>, string[]>()
{
    {
        (conf => conf.TestsType==MeasurementType.Accuracy),
        new string[] { "ozaki_accuracy.m" }
    },
    {
        (conf => conf.TestsType==MeasurementType.Performance && !conf.pinBLISThreadsToCore),
        new string[] { "ozaki_speedups.m", "ozaki_flops.m", "ozaki_efficiency.m" }
    },
    {
        (conf => conf.TestsType==MeasurementType.Performance && conf.pinBLISThreadsToCore),
        new string[] { "ozaki_speedups_pinned.m" }
    },
    {
        (conf => conf.TestsType==MeasurementType.Cycles),
        new string[] { "ozaki_cycles.m" }
    }
};




var ozaie_setup_script = "ozaie_env_setup.sh";
var measurements_folder = "measurements";
var matlab_scripts_folder = "matlab_scripts";

void AccuracyTests(TestsConfig testConfig)
{
    string sshConfigDir = testConfig.ssh_config_dir;
    bool verifyConfigs = false;
    int totalRunsInBetweenWaits = 0;
    bool writeSuccess = false;
    bool resume = testConfig.resume;

    string testsRoot = $"{testConfig.mlir_repo_dir}/programming_examples/basic/matrix_multiplication/";
    string xclBinsDir = "ozaki_xclbins";
    string testsLogFile = $@"..\..\..\{measurements_folder}\ozaki_accuracy_tests.json";
    string gnuplotDatFile = $@"..\..\..\{measurements_folder}\ozaki_accuracy_tests.dat";

    DateTime testStartTime = DateTime.Now;

    // Load the currently persisted configs 

    if (!File.Exists(testsLogFile))
    {
        File.WriteAllText(testsLogFile, "{}");
    }

    try
    {
        OzakiConfig.configs = JsonSerializer.Deserialize<List<OzakiConfig>>(File.ReadAllText(testsLogFile));
    }
    catch
    { }

    // these configs are measured so do the rest only when on resume mode
    Predicate<OzakiConfig> toSkip = c => true;
    Predicate<OzakiConfig> toForce = c =>
    {
        bool cond = c.device == Device.npu1;
        return cond;
    };

    foreach (OzakiConfig ozconf in getBiINT64SumStateConfigs(c => c.k>=testConfig.min_K && c.k<=testConfig.max_K, OzakiConfigMode.BestAccuracy))
    {
        foreach (Device dev in Enum.GetValues<Device>())
        {
            ozconf.device = dev;
            Script script = new Script(hostsMap[dev], sshConfigDir, 
                $"source {testConfig.mlir_repo_dir}/../{ozaie_setup_script}");
            string o;

            foreach (CoreConfig coreConfig in Enum.GetValues<CoreConfig>())
            {
                ozconf.coreConfig = coreConfig;
                script >>= $"cd {testsRoot}/{directoriesMap[coreConfig]}";

                int n_aie_cols = dev == Device.npu1 ? 4 : 8;

            
                var K = (int)Math.Pow(2, ozconf.k);
                var execFilenameFormat = $"{K}x{K}x{K}_16x16x16_b{ozconf.b}_d{ozconf.d}{(coreConfig == CoreConfig.whole ? $"_{n_aie_cols}c" : "")}";
                var binFile = $"final_{execFilenameFormat}.xclbin";
                var instrFile = $"insts_{execFilenameFormat}.txt";
                var logFile = $"oz_dump_K{K}_b{ozconf.b}_d{ozconf.d}{(coreConfig == CoreConfig.whole ? $"_{n_aie_cols}c" : "")}.log";

                bool? rebuildNeeded = null;

                foreach (double fi in new double[] { 0.1d, 1.0d, 2.0d, 4.0d }) {
                    ozconf.fi = fi;
                    int seedIter = 0;

                    if (rebuildNeeded.HasValue && rebuildNeeded.Value)
                    {
                        break;
                    }

                    foreach (int seed in Enumerable.Range(0, 10).Select(i => Random.Shared.Next(10000)).ToArray())
                    {
                        ozconf.seed = seed;
                        ozconf.iter = ++seedIter;

                        OzakiConfig currentRun = OzakiConfig.GetConfig(ozconf);              

                        string configStr = $"{(coreConfig == CoreConfig.whole ? $"n_aie_cols={n_aie_cols}" : "")} bps={ozconf.b} num_splits={ozconf.d} K={K}";


                        // dont run again if meant to be skipped or already successful provided not forced to run again
                        if (
                            !toForce(currentRun) 
                            && 
                            (toSkip(currentRun) || (resume && currentRun.measureSuccess)))
                        {
                            Console.WriteLine($"Skipping iter={ozconf.iter} fi={ozconf.fi} {configStr}");
                            continue;
                        }

                        // only set measure time if not skipped
                        currentRun.measureTime = DateTime.Now;

                        //################# RUN ###########################
                        //#################################################

                        if (!rebuildNeeded.HasValue || rebuildNeeded.Value) { 

                            o = script >>= $"ls -al {xclBinsDir}";
                            rebuildNeeded = !rebuildNeeded.HasValue || !o.Contains(binFile) || !o.Contains(instrFile);

                            if (rebuildNeeded.Value)
                            {
                                o = script >>= $"make clean; make {configStr} > lastrun.log";
                                o = script >>= $"ls -al {xclBinsDir}";

                                currentRun.buildStatus = o.Contains(binFile) && o.Contains(instrFile);
                                rebuildNeeded = !currentRun.buildStatus;
                            }
                        }

                        // if binaries are available either due to a rebuild now or previously
                        if (!rebuildNeeded.Value)
                        {
                            o = script >>= $"rm {logFile}";
                            Console.WriteLine($"Runnning iter#{ozconf.iter}");
                            o = script |= $"make {(coreConfig == CoreConfig.whole ? $"n_aie_cols={n_aie_cols}" : "")} bps={ozconf.b} num_splits={ozconf.d} K={K} fi={fi} seed={seed} verify=1 runspecific";
                            totalRunsInBetweenWaits++;

                            o = script >>= $"tail -n 200 {logFile}";

                            Match[] relErrorMatches = new Regex(@"(?<variant>[A-Za-z_]+)\sMaximum.relative.error.\s+(?<relativeError>.+)$", RegexOptions.Multiline).Matches(o).ToArray();

                            currentRun.measureSuccess = (relErrorMatches.Count() == Enum.GetValues<Variant>().Count());

                            relErrorMatches.ToList().ForEach(m =>
                            {
                                currentRun.maxRelativeError[Enum.Parse<Variant>(m.Groups["variant"].Value)] = double.Parse(m.Groups["relativeError"].Value);
                            });

                            currentRun.KernelTimedOut = o.Contains("kernel did not complete", StringComparison.OrdinalIgnoreCase);
                        }

                        writeSuccess = false;

                        while (!writeSuccess)
                        {
                            try
                            {
                                File.WriteAllText(testsLogFile, JsonSerializer.Serialize(OzakiConfig.configs, jsonOptions));
                                writeSuccess = true;
                                Thread.Sleep(1000);
                            }
                            catch (IOException e)
                            {

                            }
                        }

                        // if rebuild is still needed, it means it failed, so break out to another configuration
                        if (rebuildNeeded.Value)
                        {
                            break;
                        }

                        try
                        {
                            Thread.Sleep(pauseBetweenTestMeasurementsMillis);
                        }
                        catch { }

                        // if kernel timed out, break to next fi configuration
                        // and not unnecessarily repeat the seed runs
                        if (currentRun.KernelTimedOut)
                        {
                            break;
                        }
                    }

                    writeSuccess = false;

                    while (!writeSuccess)
                    {
                        try
                        {
                            File.WriteAllText(testsLogFile, JsonSerializer.Serialize(OzakiConfig.configs, jsonOptions));
                            OzakiConfig[] batch =
                                OzakiConfig.GetMatchingConfigs(ozconf, new string[] { nameof(ozconf.iter) })
                                .Where(c => c.measureTime.CompareTo(testStartTime)>0 && c.maxRelativeError.Count == 3)
                                .ToArray();


                            if (batch.Count() > 0)
                            {
                                File.AppendAllText(gnuplotDatFile, string.Format("{0,4} {1,8} {2,2} {3,2} {4} {5,2} {6} {7} {8} {9}\n",
                                        ozconf.device,
                                        ozconf.coreConfig,
                                        ozconf.fi,
                                        ozconf.k,
                                        ozconf.b,
                                        ozconf.l_mant,
                                        batch.Average(c => c.maxRelativeError[Variant.Ozaki]),
                                        batch.Average(c => c.maxRelativeError[Variant.DGEMM]),
                                        batch.Average(c => c.maxRelativeError[Variant.BLIS_DGEMM]),
                                        batch.Count()
                                    )
                                );
                            }

                            writeSuccess = true;
                            Thread.Sleep(1000);
                        }
                        catch (IOException e)
                        {

                        }
                    }
                }              

                try
                {                 

                    if (totalRunsInBetweenWaits > 10)
                    {
                        Task.Delay(30000).Wait();
                        totalRunsInBetweenWaits = 0;
                    }
                }
                catch (Exception e) { }
            }

            script.Disconnect();
        }

    }
    
    GeneratePlots(testConfig);
}

void CyclesMeasurements(TestsConfig testConfig)
{
    string sshConfigDir = $@"C:\cygwin64\home\malik\.ssh";
    bool verifyConfigs = false;
    int totalRunsInBetweenWaits = 0;

    string testsRoot = $"{testConfig.mlir_repo_dir}/programming_examples/basic/matrix_multiplication/";
    string testsLogFile = $@"..\..\..\{measurements_folder}\ozaki_cycles.json";
    string gnuplotDatFile = $@"..\..\..\{measurements_folder}\ozaki_cycles.dat";
    string traceFile = @"trace_mm.json";
    string remoteLastRunLog = "lastrun.log";
    bool writeSuccess = false;
    bool resume = testConfig.resume;
    

    File.WriteAllText(gnuplotDatFile, "");

    if (!File.Exists(testsLogFile))
    {
        File.WriteAllText(testsLogFile, "{}");
    }

    try
    {
        OzakiConfig.configs = JsonSerializer.Deserialize<List<OzakiConfig>>(File.ReadAllText(testsLogFile));
    }
    catch
    { }


    foreach (Device dev in Enum.GetValues<Device>())
    {
        OzakiConfig conf = new OzakiConfig();
        conf.device = dev;
        Script script = new Script(hostsMap[dev], sshConfigDir,
            $"source {testConfig.mlir_repo_dir}/../{ozaie_setup_script}");
        string o;
        script >>= $"cd {testsRoot}/{directoriesMap[CoreConfig.single]}";

        string[] kernelFiles = new string[] {
            "../../../../aie_kernels/aie2/mm_ozaki.cc",
            "../../../../aie_kernels/aie2p/mm_ozaki.cc",
            "../../../../aie_kernels/generic/ozaki_base.cc"
        };

        string[] AMDKernelFiles = new string[] {
            "../../../../aie_kernels/aie2/mm.cc",
            "../../../../aie_kernels/aie2p/mm.cc",
            "../../../../aie_kernels/aie2/zero.cc",
            "../../../../aie_kernels/aie2p/zero.cc",
        };

        foreach (int b in new int[] { 4, 5, 6, 7 })
        {
            conf.b = b;

            foreach (int l_mant in new int[] { 50, 60, 70, 80 })
            {
                conf.l_mant = l_mant;

                foreach (int k in new int[] { 8 } )
                {
                    conf.k = k;
                    var K = (int)Math.Pow(2, conf.k);
                    OzakiConfig currentRun = OzakiConfig.GetConfig(conf);

                    foreach (PhasesTags phase in Enum.GetValues<PhasesTags>())
                    {
                        string configStr = $"bps={currentRun.b} num_splits={currentRun.d} K={K}";
                        currentRun.extraInfo += phase;

                        // dont measure again
                        if (resume)
                        {
                            if (currentRun.phaseCyclesCount.ContainsKey(phase))
                            {
                                Console.WriteLine($"Skipping {configStr} phase:{phase.ToString()}");                                
                                continue;
                            }
                        }

                        configStr += $" {currentRun.extra_build_options}";

                        string line_identifier_middle = "_cycles_measure_(stop|start)";

                        string uncommentCmd = $@"sed -i -E '/<phase_name>{line_identifier_middle}/s@/\*[[:space:]]*(event0|event1)\(\);[[:space:]]*\*/@\1();@'";
                        string commentCmd = $@"sed -i -E '/<phase_name>{line_identifier_middle}/s@^([[:space:]]*)(event0|event1)\(\);@\1/* \2(); */@'";

                        // comment all event0 and event1 lines before proceeding
                        script >>= $"{commentCmd.Replace("<phase_name>", "")} {string.Join(" ", kernelFiles)}";
                        // also command event0 and event1 in the original mm.cc and zero.cc from AMD
                        script >>= $"sed -i -E '/^[[:space:]]*event[01]\\(\\);[[:space:]]*$/s#^#//#' {(string.Join(" ", AMDKernelFiles))}";

                        // uncomment event0 and event1 line for the current phase in kernel code
                        script >>= $"{uncommentCmd.Replace("<phase_name>",phase.ToString())} {string.Join(" ", kernelFiles)}";                      

                        o = script >>= $"rm {traceFile}";
                        o = script >>= $"make clean; make {configStr} trace > {remoteLastRunLog}";

                        o = script >>= $"ls -al *.json";

                        Thread.Sleep(5000);

                        currentRun.buildStatus = o.Contains(traceFile);                     

                        // comment all event0 and event1 lines
                        script >>= $"{commentCmd.Replace("<phase_name>", "")} {string.Join(" ", kernelFiles)}";

                        if (currentRun.buildStatus)
                        {
                            o = script >>= $"make {configStr} parse_trace";

                            Match callsCountMatch = new Regex(@".+?invocations.+?(?<call_count>\d+)", RegexOptions.Multiline).Match(o);
                            Match cyclesMatch = new Regex(@"\d+\/\s+(?<Min>\d+)\/\s+(?<Avg>(\d|\.)+)\/\s+(?<Max>\d+)").Match(o);

                            if (cyclesMatch.Success)
                            {
                                currentRun.phaseCyclesCount[phase] = Convert.ToInt32(float.Parse(cyclesMatch.Groups[Aggregate.Avg.ToString()].Value));
                                currentRun.phaseCallsCount[phase] = int.Parse(callsCountMatch.Groups["call_count"].Value);
                                currentRun.measureSuccess = true;
                            }                       
                        }
                        else
                        {
                            o = script >>= $"tail -n 20 {remoteLastRunLog}";
                            currentRun.buildLogTail = o;
                            currentRun.KernelTimedOut = o.Contains("kernel did not complete", StringComparison.OrdinalIgnoreCase);
                            
                            // this clause only to verify if we skip a build trace json due to it building slow
                            if (!currentRun.KernelTimedOut && currentRun.buildLogTail.Contains($"--output {traceFile}"))
                            {
                                Thread.Sleep(60000);
                                o = script >>= $"ls -al *.json";
                                if (o.Contains(traceFile))
                                {
                                    currentRun.buildLogTail += "|mistake in finding trace json file";
                                }

                            }
                        }

                        writeSuccess = false;

                        while (!writeSuccess)
                        {
                            try
                            {
                                File.WriteAllText(testsLogFile, JsonSerializer.Serialize(OzakiConfig.configs, jsonOptions));
                                writeSuccess = true;
                                //File.Copy()
                            }
                            catch (IOException e)
                            {

                            }
                        }

                        try
                        {                           
                            if (totalRunsInBetweenWaits > 10)
                            {
                                Task.Delay(60000).Wait();
                                totalRunsInBetweenWaits = 0;
                            } else
                            {
                                Thread.Sleep(pauseBetweenTestMeasurementsMillis);
                            }
                        }
                        catch (Exception e)
                        { }

                    }

                    writeSuccess = false;

                    while (!writeSuccess)
                    {
                        try
                        {
                            File.WriteAllText(testsLogFile, JsonSerializer.Serialize(OzakiConfig.configs, jsonOptions));
                            if (currentRun.measureSuccess)
                            {
                                File.AppendAllText(gnuplotDatFile, string.Format("{0,4} {1,2} {2,2} {3,2} {4,2} {5,5} {6,5} {7,5} {8,5} {9,5} {10,5} {11,5} {12,5} {13,5} {14,5} {15,5} {16,5}\n",
                                        currentRun.device,
                                        currentRun.b,
                                        currentRun.l_mant,
                                        currentRun.d,
                                        currentRun.k,
                                        currentRun.phaseCyclesCount.GetValueOrDefault(PhasesTags.max_exp_a),
                                        currentRun.phaseCallsCount.GetValueOrDefault(PhasesTags.max_exp_a),
                                        currentRun.phaseCyclesCount.GetValueOrDefault(PhasesTags.max_exp_b),
                                        currentRun.phaseCallsCount.GetValueOrDefault(PhasesTags.max_exp_b),
                                        currentRun.phaseCyclesCount.GetValueOrDefault(PhasesTags.splits_a),
                                        currentRun.phaseCallsCount.GetValueOrDefault(PhasesTags.splits_a),
                                        currentRun.phaseCyclesCount.GetValueOrDefault(PhasesTags.splits_b),
                                        currentRun.phaseCallsCount.GetValueOrDefault(PhasesTags.splits_b),
                                        currentRun.phaseCyclesCount.GetValueOrDefault(PhasesTags.muls),
                                        currentRun.phaseCallsCount.GetValueOrDefault(PhasesTags.muls),
                                        currentRun.phaseCyclesCount.GetValueOrDefault(PhasesTags.sum),
                                        currentRun.phaseCallsCount.GetValueOrDefault(PhasesTags.sum)
                                    )
                                );
                            }
                            writeSuccess = true;                           
                        }
                        catch (IOException e)
                        {
                            Thread.Sleep(1000);
                        }
                    }

                }

            }
        }

        script.Disconnect();
    }

    GeneratePlots(testConfig);
}


void PerformanceTests(TestsConfig testConfig)
{
    string sshConfigDir = testConfig.ssh_config_dir;
    bool verifyConfigs = false;
    int totalRunsInBetweenWaits = 0;
    const int BLIS_THREADS = 0;

    string testsRoot = $"{testConfig.mlir_repo_dir}/programming_examples/basic/matrix_multiplication/";
    string xclBinsDir = "ozaki_xclbins";
    string testsLogFile = $@"..\..\..\{measurements_folder}\ozaki_perf_tests{(testConfig.pinBLISThreadsToCore?"_pinned":"")}.json";
    string gnuplotDatFile = $@"..\..\..\{measurements_folder}\ozaki_perf_tests{(testConfig.pinBLISThreadsToCore ? "_pinned" : "")}.dat";
    bool resume = testConfig.resume;

    // Load the currently persisted configs 

    if (!File.Exists(testsLogFile))
    {
        File.WriteAllText(testsLogFile, "{}");
    }

    try
    {
        OzakiConfig.configs = JsonSerializer.Deserialize<List<OzakiConfig>>(File.ReadAllText(testsLogFile));
    }
    catch
    { }

    List<OzakiConfig> configsToRun = new List<OzakiConfig>();

    foreach (Device dev in Enum.GetValues<Device>().ToArray())
    {
        foreach (OzakiConfig ozconf in getBiINT64SumStateConfigs(c => c.l_mant == testConfig.l_mant && (c.k >= testConfig.min_K && c.k <= testConfig.max_K), OzakiConfigMode.BestPerformance))
        {
            ozconf.device = dev;

            foreach (CoreConfig coreConfig in Enum.GetValues<CoreConfig>())
            {
                ozconf.coreConfig = coreConfig;
                int[] aieColsRange = coreConfig == CoreConfig.single ?
                    new int[] { 0 } : new int[] { 1, 2, 4 }.Concat(dev == Device.npu2 ? new int[] { 8 } : new int[] { }).ToArray();

                foreach (int n_aie_cols in aieColsRange)
                {
                    ozconf.NumAIEColumns = n_aie_cols;

                    // for BLIS threads pinned configuration, we do not want to run more threads than
                    // there are physical cpu cores so that each thread has exclusive access to core's resources
                    if (!(testConfig.pinBLISThreadsToCore && (ozconf.NumAIEColumns*4) > cpuCoresMap[dev]))
                    {
                        ozconf.NumBLISThreads = coreConfig == CoreConfig.single ? 1 : (n_aie_cols * 4);
                        configsToRun.Add(ozconf.Clone());
                    }               
                }

                // in this tests configuration where we pin BLIS threads to cores
                // we also want to compare:
                // 1) a whole-array Oz-AIE against All-CPU-core BLIS run
                // 2) a whole-array Oz-AIE against single-core BLIS run
                if (testConfig.pinBLISThreadsToCore)
                {
                    if (ozconf.coreConfig == CoreConfig.single)
                    {
                        ozconf.NumBLISThreads = cpuCoresMap[dev];
                        configsToRun.Add(ozconf.Clone());
                    }
                    else if (ozconf.coreConfig == CoreConfig.whole && ozconf.NumAIEColumns == aieColsRange.Last())
                    {
                        ozconf.NumBLISThreads = 1;
                        configsToRun.Add(ozconf.Clone());
                        ozconf.NumBLISThreads = cpuCoresMap[dev];
                        configsToRun.Add(ozconf.Clone());
                    }
                }
            }
        }
    }

    OzakiConfig prevRun = null;
    Script script = null;

    foreach (OzakiConfig ozconf in configsToRun)
    {
        if (prevRun?.device != ozconf.device)
        {
            // only disconnect if the device has changed
            if (prevRun!=null)
            {
                script.Disconnect();
            }

            script = new Script(hostsMap[ozconf.device], sshConfigDir,
                $"source {testConfig.mlir_repo_dir}/../{ozaie_setup_script}");

            // move to the single-core or whola-array directory
            script >>= $"cd {testsRoot}/{directoriesMap[ozconf.coreConfig]}";
        }
        string o;


        int[] aieColsRange = ozconf.coreConfig == CoreConfig.single ?
            new int[] { 0 } : new int[] { 1, 2, 4 }.Concat(ozconf.device == Device.npu2 ? new int[] { 8 } : new int[] { }).ToArray();

        string blis_thread_config = "";     

        var K = (int)Math.Pow(2, ozconf.k);
        var execFilenameFormat = $"{K}x{K}x{K}_16x16x16_b{ozconf.b}_d{ozconf.d}{(ozconf.coreConfig == CoreConfig.whole ? $"_{ozconf.NumAIEColumns}c" : "")}";
        var binFile = $"final_{execFilenameFormat}.xclbin";
        var instrFile = $"insts_{execFilenameFormat}.txt";
        var logFile = $"oz_dump_K{K}_b{ozconf.b}_d{ozconf.d}{(ozconf.coreConfig == CoreConfig.whole ? $"_{ozconf.NumAIEColumns}c" : "")}.log";

        bool? rebuildNeeded = null;

        OzakiConfig currentRun = OzakiConfig.GetConfig(ozconf);

        // dont run again if already successful
        if (resume && currentRun.measureSuccess)
        {
            Console.WriteLine($"Skipping {binFile}");
            prevRun = currentRun;
            continue;
        }

        script >>= $"cd {testsRoot}/{directoriesMap[ozconf.coreConfig]}";

        // only set measure time if not skipped
        currentRun.measureTime = DateTime.Now;

        //################# RUN ###########################
        //#################################################
        int iters = 10;
        int warmup = 10;
        int iters_dgemm = 10;
        int warmup_dgemm = 10;
        int iters_blis = 100;
        int warmup_blis = 200;
        bool cold_cache_blis = false;

        // set the BLIS thread count
        blis_thread_config = $"BLIS_NUM_THREADS={(BLIS_THREADS > 0 ? BLIS_THREADS : ozconf.NumBLISThreads)}" +
                    (
                    testConfig.pinBLISThreadsToCore ?
                    $" GOMP_CPU_AFFINITY=\"{string.Join(" ", cpu_affinity[ozconf.device].Take(ozconf.NumBLISThreads))}\"" :
                    "");


        if (!currentRun.forceNoRebuild && (!rebuildNeeded.HasValue || rebuildNeeded.Value))
        {
            o = script >>= $"ls -al {xclBinsDir}";
            rebuildNeeded = !o.Contains(binFile) || !o.Contains(instrFile);

            if (rebuildNeeded.Value)
            {
                o = script >>= $"make clean; {blis_thread_config} make {(ozconf.coreConfig == CoreConfig.whole ? $"n_aie_cols={ozconf.NumAIEColumns}" : "")} " +
                                $"bps={ozconf.b} num_splits={ozconf.d} K={K} > lastrun.log";

                o = script >>= $"ls -al {xclBinsDir}";

                currentRun.buildStatus = o.Contains(binFile) && o.Contains(instrFile);
                rebuildNeeded = !currentRun.buildStatus;
            }
        }

        // if binaries are available either due to a rebuild now or previously
        if (currentRun.forceNoRebuild || !rebuildNeeded.Value)
        {
            o = script >>= $"rm {logFile}";

            o = script |= $"{blis_thread_config} make {(ozconf.coreConfig == CoreConfig.whole ? $"n_aie_cols={ozconf.NumAIEColumns}" : "")} " +
                $"cold_cache_blis={cold_cache_blis.ToString().ToLower()} " +
                $"iters={iters} warmup={warmup} iters_dgemm={iters_dgemm} warmup_dgemm={warmup_dgemm} iters_blis={iters_blis} warmup_blis={warmup_blis} " +
                $"bps={ozconf.b} num_splits={ozconf.d} K={K} verify={(currentRun.doVerify? 1 : 0)} runspecific";
            totalRunsInBetweenWaits++;

            o = script >>= $"tail -n 100 {logFile}";

            Match[] matmulTimeMatches = new Regex(@"(?<aggregate>(Min)|(Max)|(Avg)|(Median))\s((?<variant>[A-Za-z_]+)).+?(?<time>[\d\.e\-\+]+)us.*?$", RegexOptions.Multiline).Matches(o).ToArray();
            Match[] flopsMatches = new Regex(@"(?<aggregate>(Min)|(Max)|(Avg)|(Median)).+?(gflops).+?(?<flops>[\d\.e\-\+]+)", RegexOptions.Multiline).Matches(o).ToArray();
            Match relErrorMatch = new Regex(@"Ozaki Maximum.relative.error.\s+(?<relativeError>.+)$", RegexOptions.Multiline).Match(o);

            currentRun.measureSuccess = matmulTimeMatches.Count() >= Enum.GetValues<Variant>().Count();

            foreach (Match m in matmulTimeMatches.Concat(flopsMatches).ToArray())
            {
                if (m.Groups.ContainsKey("time"))
                {
                    currentRun.runtimeMicrosecs[Enum.Parse<Variant>(m.Groups["variant"].Value)][Enum.Parse<Aggregate>(m.Groups["aggregate"].Value)] = double.Parse(m.Groups["time"].Value);
                }

                if (m.Groups.ContainsKey("flops"))
                {
                    currentRun.GFLOPS[Enum.Parse<Aggregate>(m.Groups["aggregate"].Value)] = double.Parse(m.Groups["flops"].Value);
                }
            }

            if (relErrorMatch.Success)
            {
                currentRun.maxRelativeError[Variant.Ozaki] = double.Parse(relErrorMatch.Groups["relativeError"].Value);
            }

            currentRun.KernelTimedOut = o.Contains("kernel did not complete");
        }

        bool writeSuccess = false;

        while (!writeSuccess)
        {
            try
            {
                File.WriteAllText(testsLogFile, JsonSerializer.Serialize(OzakiConfig.configs, jsonOptions));
                if (currentRun.measureSuccess)
                {
                    File.AppendAllText(gnuplotDatFile, string.Format("{0,4} {1,2} {2,1} {3,2} {4, 20} {5, 20} {6, 20} {7, 20} {8, 20}\n",
                            currentRun.device,
                            currentRun.k,
                            currentRun.NumAIEColumns,
                            currentRun.NumBLISThreads,
                            currentRun.runtimeMicrosecs[Variant.Ozaki][Aggregate.Median],
                            currentRun.runtimeMicrosecs[Variant.DGEMM][Aggregate.Median],
                            currentRun.runtimeMicrosecs[Variant.BLIS_DGEMM][Aggregate.Median],
                            currentRun.runtimeMicrosecs[Variant.NPU][Aggregate.Median],
                            currentRun.runtimeMicrosecs[Variant.CPU][Aggregate.Median]
                        )
                    );
                }
                writeSuccess = true;                            
            }
            catch (IOException e)
            {
                Thread.Sleep(1000);
            }
        }

        try
        {
                        
            if (totalRunsInBetweenWaits > 5)
            {
                Task.Delay(60000).Wait();
                totalRunsInBetweenWaits = 0;
            } 
            else
            {
                Thread.Sleep(pauseBetweenTestMeasurementsMillis);
            }
        }
        catch (Exception e)
        { }

        prevRun = currentRun;
    }

    GeneratePlots(testConfig);
}

void GeneratePlots(TestsConfig testConfig)
{
    string[] scripts = testTypeToScriptsMap[testTypeToScriptsMap.Keys.First(m => m(testConfig))];

    scripts.ToList().ForEach(script =>
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = testConfig.matlab_bin,
            Arguments = $"-sd \"..\\..\\..\\{matlab_scripts_folder}\" -batch \"run('{script}');\"",
            UseShellExecute = false
        };

        var p = Process.Start(startInfo);
        p.WaitForExit();
    });
}



static List<OzakiConfig> getBiINT64SumStateConfigs(Predicate<OzakiConfig> selectionClause, OzakiConfigMode configMode, int topN = 1)
{
    var b_range = Enumerable.Range(3, 13);
    var l_mant_range = Enumerable.Range(53, 87);
    var k_range = Enumerable.Range(7, 8);

    List<OzakiConfig> grid = new List<OzakiConfig>();

    Func<int, int, int, int, int, int> lengthOfAccumulation = (b, l_mant, k, p64, p64prev) =>
    {
        var d = (l_mant + (b - 1)) / b;
        var binWiseSumBitsInStartingBin = d - p64prev;

        return b * (p64 + 1) + k + Convert.ToInt32(Math.Ceiling(Math.Log2(binWiseSumBitsInStartingBin - (p64 - 1)))) + (p64 == 1 ? 0 : 1);
    };

    var tmp = b_range.Select(b =>
    {
        l_mant_range.Select(l_mant =>
        {
            var d = (l_mant + (b - 1)) / b;

            k_range.Select(k =>
            {
                int binNo = 1;
                int binsInPrevINT64s = 0;
                int prevP64 = 0;

                int currLen = 0;
                int prevLen = 0;

                List<(int, int)> lengths = new List<(int, int)>();

                for (; binNo <= d; binNo++)
                {
                    currLen = lengthOfAccumulation(b, l_mant, k, binNo - binsInPrevINT64s, binsInPrevINT64s);

                    if (currLen > 63)
                    {
                        if (binNo > 1)
                        {
                            lengths.Add((binNo - 1 - binsInPrevINT64s, prevLen));
                            binsInPrevINT64s = binNo - 1;
                            binNo--;
                        }
                    }

                    prevLen = currLen;
                }

                lengths.Add((d - binsInPrevINT64s, currLen));

                var throughput = b switch
                {
                    3 => 8,
                    > 3 and < 8 => 4,
                    > 8 => 1,
                    _ => 1
                };
                var n_muls = d * (d + 1) / 2;


                grid.Add(new OzakiConfig(b, l_mant, k, lengths));

                return 0;
            }).ToArray();

            return 0;
        }).ToArray();
        return 0;
    }).ToArray();

    var grid1 = grid.ToList();

    grid1 = grid1.Where(t => t.n_int64s == 2 && selectionClause(t)).ToList();

    if (configMode == OzakiConfigMode.BestPerformance)
    {
        grid1 = grid1.OrderBy(t => t.l_mant * 100000 + t.k * 1000 + ((float)t.n_muls / (float)t.throughput)).ToList();
        grid1 = grid1.GroupBy(t => (t.l_mant, t.k)).SelectMany(g => g.Take(topN).ToArray()).ToList();
    }
    else if (configMode == OzakiConfigMode.BestAccuracy)
    {
        grid1 = grid1.OrderBy(t => t.k * 100000 + t.l_mant * 1000).ToList();
        grid1 = grid1.GroupBy(t => t.k).SelectMany(g => g.Reverse().Take(topN).ToArray()).ToList();
    }

    string grid1Text = string.Join("\n", grid1.Select(l => l == null ? "null" : l.ToString()));

    Console.WriteLine(grid1Text);

    Console.WriteLine("Count:{0}", grid1.Count());

    return grid1;

}
    
string testConfigFile = @"..\..\..\tests_conf.json";
TestsConfig testConfig = JsonSerializer.Deserialize<TestsConfig>(File.ReadAllText(testConfigFile));

switch (testConfig.TestsType)
{
    case MeasurementType.Performance:
        PerformanceTests(testConfig);
        break;
    case MeasurementType.Accuracy:
        AccuracyTests(testConfig);
        break;
    case MeasurementType.Cycles:
        CyclesMeasurements(testConfig);
        break;
    default:
        break;
}

    
