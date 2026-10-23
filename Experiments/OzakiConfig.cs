// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
using Org.BouncyCastle.Bcpg;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace ozAIE.Experiments
{
    public enum MemApproach
    {
        two_calls,
        d_plus_one_calls,
        splits_every_mul
    }

    public enum Device
    {
        npu1,
        npu2
    }

    public enum CoreConfig
    {
        single,
        whole
    }

    public enum Variant
    {
        Ozaki,
        BLIS_DGEMM,
        DGEMM,
        NPU,
        CPU
    }

    public enum Aggregate
    {
        Avg,
        Min,
        Max,
        Median
    }

    public enum PhasesTags
    {
        max_exp_a,
        max_exp_b,
        splits_a,
        splits_b,
        muls,
        sum
    }

    public class OzakiConfig 
    {
        public static List<OzakiConfig> configs = new List<OzakiConfig>();
        private List<string> updatedProps = new List<string>();
        // NPU1 or NPU2
        [RunAttribute]
        public Device device { get; set; }
        [RunAttribute]
        public CoreConfig coreConfig { get; set; }
        [RunAttribute]
        public double fi { get; set; }
        public int seed { get; set; }
        [RunAttribute]
        public int iter { get; set; }
        private int _n_aie_cols = 4;
        [RunAttribute]
        public int NumAIEColumns { get => coreConfig == CoreConfig.single ? 0 : _n_aie_cols; set { _n_aie_cols = value; } }
        [RunAttribute]
        public int NumBLISThreads { get; set; }

        public bool KernelTimedOut { get; set; }

        /// <summary>
        /// Bits per slice
        /// </summary>
        [RunAttribute]
        public int b { get; set; }
        /// <summary>
        /// Mantissa length to cover
        /// </summary>
        public int l_mant { get; set; }
        /// <summary>
        /// log2(K) as in M x K x N matrix multiplication
        /// </summary>
        [RunAttribute]
        public int k { get; set; }
        /// <summary>
        /// Number of splits to conver l_mant
        /// </summary>
        [RunAttribute]
        public int d {
            get {
                try
                {
                    return (l_mant + (b - 1)) / b;
                }
                catch (DivideByZeroException e)
                {
                    return 0;
                }
            }
        }

        public bool buildStatus { get; set; }
        public string buildLogTail { get; set; }
        public bool doVerifyConfig { get; set; }
        public bool verifySuccess { get; set; }
        public bool measureSuccess { get; set; }
        public bool forceNoRebuild;
        public bool doVerify = true;

        public string extraInfo { get; set; }

        public DateTime measureTime { get; set; }

        public Dictionary<PhasesTags, int> phaseCyclesCount { get; set; }
        public Dictionary<PhasesTags, int> phaseCallsCount { get; set; }

        public Dictionary<Variant, double> maxRelativeError { get; set; }

        public Dictionary<Variant, Dictionary<Aggregate, double>> runtimeMicrosecs { get; set; }
        public Dictionary<Aggregate, double> GFLOPS { get; set; }

        public List<(int, int)> i64s { get; set; }

        // tile size 16x16 or 32x32
        public int m = 16;

        public OzakiConfig() 
        {
            runtimeMicrosecs = new Dictionary<Variant, Dictionary<Aggregate, double>>();
            
            foreach (Variant v in Enum.GetValues<Variant>()) { runtimeMicrosecs[v] = new Dictionary<Aggregate, double>(); };

            phaseCyclesCount = new Dictionary<PhasesTags, int>();
            phaseCallsCount = new Dictionary<PhasesTags, int>();
            maxRelativeError = new Dictionary<Variant, double>();
            GFLOPS = new Dictionary<Aggregate, double>();
        }

        public OzakiConfig(int ab, int  al_mant, int ak, List<(int,int)> a_i64s) : this()
        {
            b = ab;
            l_mant = al_mant;
            k = ak;
            i64s = a_i64s;

            var tilesMemBits = 6 * m * m * 64;
            var stackKB = 3;
            var maxExpMemBits = 2 * 16 * m;
            var binsMemBits = d * AccumBits * m * m;

            foreach (MemApproach memApproach in Enum.GetValues(typeof(MemApproach)).OfType<MemApproach>())
            {
                int fac_splits = memApproach switch
                {
                    MemApproach.two_calls => 2 * d,
                    MemApproach.d_plus_one_calls => d + 1,
                    MemApproach.splits_every_mul => 2
                };

                var splitsMemBits = fac_splits * l_int * m * m;

                Mem[memApproach] = (float)(tilesMemBits + maxExpMemBits + binsMemBits + splitsMemBits + ExtraInt64MemBitsForC) / (float)(8 * 1024);
            }
        }

        public string extra_build_options = "";

        public int throughput { get =>
                b switch
                {
                    3 => AccumBits == 32 ? 8 : 2,
                    > 3 and < 8 => AccumBits==32? 4 : 2,
                    >= 8 => 1,
                    _ => 1
                };
        }

        public int l_int
        {
            get =>
                b switch
                {

                    3 => 4,
                    > 3 and < 8 => 8,
                    >= 8 => 16,
                    _ => 8
                };
        }

        public int n_muls { get => d * (d + 1) / 2; }

        public int AccumBits { get => (31 - (2 * b) - Convert.ToInt32(Math.Ceiling(Math.Log2(d)))) >= k ? 32 : 64; }   

        public Dictionary<MemApproach, float> Mem = new Dictionary<MemApproach, float>();

        public int n_int64s { 
            get 
            { 
                if (i64s==null)
                {
                    return 0;
                }
                return i64s.Count() + (i64s.Last().Item2 <= (64 - 13) ? 0 : 1); 
            } 
        }

        public float ExtraInt64MemBitsForC { 
            // (n_int64s - 1) . subtract 1 because the original C tile already is 1 INT64 per element
            // m^2 is the number of INT64 elements for C (or C tile size)
            // 64 - for 64-bit elements
            // 2 - ping-pong factor for buffer space of C
            get => (n_int64s - 1) * (m * m) * 64 * 2;  }

        public override string ToString()
        {
            return string.Format("{0,5},{1,5},{2,5},{3,5},{4,5},{5,7},{6,7}, {7,5},{8,5},{9,5},{10,5}",
            b,
            l_mant,
            k,
            d,
            throughput,
            n_muls,
            ((float)n_muls / (float)throughput),
            "(" + string.Join(",", i64s.Select(t => t.Item1 + ":" + t.Item2)) + ")",
            string.Format("({0})", string.Join(",",
                new float[] {
                    Mem[MemApproach.two_calls],
                    Mem[MemApproach.d_plus_one_calls],
                    Mem[MemApproach.splits_every_mul]
                })),
            n_int64s,
            AccumBits
            );
        }

        public string serialize()
        {
            JsonSerializerOptions options = new JsonSerializerOptions();
            options.IncludeFields = true;
            return JsonSerializer.Serialize(this, typeof(OzakiConfig));
        }

        public int GetMinimumBinsInFirstINT64ForBiINT64SumState()
        {
            int numerator = b * (d + 1) - 51 + k;

            return numerator < 0? 1: (numerator / b + 1);
        }
        
        public OzakiConfig Clone()
        {
            return JsonSerializer.Deserialize<OzakiConfig>(JsonSerializer.Serialize<OzakiConfig>(this));
        }

        public static OzakiConfig GetConfig(OzakiConfig refConfig)
        {
            PropertyInfo[] runProperties = typeof(OzakiConfig).GetProperties(BindingFlags.Instance | BindingFlags.Public)
                .Where(pi => pi.GetCustomAttribute<RunAttribute>() != null).ToArray();

            OzakiConfig result;

            OzakiConfig fromList = configs.SingleOrDefault(c => runProperties.All(pi => pi.GetValue(c).Equals(pi.GetValue(refConfig))));

            if (fromList != null)
            {
                result = fromList;
            }
            else
            {
                result = refConfig.Clone();
                configs.Add(result);
            }

            result.extraInfo = "";
            return result;
        }

        public static OzakiConfig[] GetMatchingConfigs(OzakiConfig refConfig, string[] propsToExcludeFromMatch)
        {
            PropertyInfo[] runProperties = typeof(OzakiConfig).GetProperties(BindingFlags.Instance | BindingFlags.Public)
                .Where(pi => !propsToExcludeFromMatch.Contains(pi.Name))
                .Where(pi => pi.GetCustomAttribute<RunAttribute>() != null).ToArray();

            return configs.Where(c => runProperties.All(pi => pi.GetValue(c).Equals(pi.GetValue(refConfig)))).ToArray();
        }

    }
}
