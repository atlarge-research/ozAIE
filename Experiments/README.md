Experiment and Plotting Dependencies
Windows 11
Visual Studio Community 2022, version 17.14 or later
.NET 8.0 SDK
MATLAB R2023b
Getting Started

To collect measurements for the accuracy and performance tests, the following libraries should be installed.
MPFR 4.2.2 
AOCL-BLIS 5.0.0 (with OpenMP-enabled configuration)

The compilation of the measurments launcher file inside the remote installation (on XDNA1 and XDNA2 platforms) depends on pkg-config to link against the header and library files of the above-mentioned dependencies. pkg-config should be available on the remote machine's OS. 

To run the experiments, a Windows machine is required with Visual Studio Community 2022 and .NET 8.0 SDK installed. To build the experiments project using Visual Studio, open the Tools menu inside Visual Studio, select the Command Line submenu, and then select Developer Command Prompt to open the developer shell. To build the experiments executable:

```Windows
C:\> cd OzAIE/Experiments
C:\> dotnet build Experiments.csproj -f net8.0
```

The executable experiment uses SSH to connect remotely to the XDNA1 and XDNA2 platforms and execute the measurements. SSH configuration of both remote hosts is required with public-private key-based authentication. To configure the test configurations to run, a tests_conf.json file exists under the Experiments folder:

```json
  "TestsType": 0,
  "pinBLISThreadsToCore" : false,
  "resume": false,
  "ssh_config_dir" : "C:\\home\\user\\.ssh",
  "mlir_repo_dir": "~/mlir-aie-v1.3.2",
  "l_mant": 53,
  "min_K": 128,
  "max_K": 2048,
  "hostname_xdna1": "xdna",
  "hostname_xdna2": "xdna2",
  "matlab_bin": "C:\\MATLAB\bin\matlab.exe"
```

The table below shows the property values that can be set in the above JSON file:

| Property | Description |
|---|---|
| `TestsType` | Specifies the type of experiments to execute. `0` for Accuracy, `1` for Performance, `2` for Cycle measurements. |
| `pinBLISThreadsToCore` | Specifies whether to pin BLIS threads to physical cores. Only applicable when `TestsType` is set to `1`. |
| `resume` | Indicates whether to resume a previous run or start a new run. |
| `ssh_config_dir` | Specifies the path to the directory containing SSH configuration files on the current machine. |
| `mlir_repo_dir` | Specifies the path to the remote MLIR repository on the XDNA1 and XDNA2 machines. |
| `l_mant` | Specifies the mantissa to be covered (\(l_m\)) in the performance tests. Only applicable when `TestsType` is set to `1`. |
| `min_K` | Defines the minimum value of the matrix dimension \(K\). Should be a power of 2. |
| `max_K` | Defines the maximum value of the matrix dimension \(K\). Should be a power of 2. |
| `hostname_xdna1` | Specifies the hostname or network alias of the XDNA1 node as defined in the SSH config file. |
| `hostname_xdna2` | Specifies the hostname or network alias of the XDNA2 node as defined in the SSH config file. |
| `matlab_bin` | Specifies the full path to the MATLAB executable. |
```

To run the particular measurements (i.e, accuracy, performance or cycles) configured in the JSON file:

C:\> run_measurements.bat


Once the batch file has completed, it will invoke MATLAB to generate the relevant plots. A data file with the .dat extension will also be available in the Experiments/measurements directory, containing the measurement data used for plot generation. The plots PDFs and PNGs will be automatically saved to the Experiments/plots directory. 

