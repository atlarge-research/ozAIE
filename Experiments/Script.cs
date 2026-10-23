// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
using Org.BouncyCastle.Asn1;
using Org.BouncyCastle.Tls;
using Renci.SshNet;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace ozAIE.Experiments
{
    /// <summary>
    /// Represent the SSH Shell for executing scripts 
    /// </summary>
    internal class Script
    {
        string _host;
        string _sshConfigDir;
        SshClient _client;
        SshClient _jump;
        ShellStream _shell;
        List<string> initCommands = new List<string>();
        public string LastOutput { get; private set; } = "";
        CancellationTokenSource cancelSrc = new CancellationTokenSource();
        Task _reconnectLoop;
        bool _isShutdown = false;
        const string _prompt = "__SSH_PROMPT__";
        string _pwd = null;

        string ReadUntil(bool storeOutput = true)
        { 
            var output = new StringBuilder();

            while (true)
            {
                if (_shell.DataAvailable)
                {
                    output.Append(_shell.Read());                

                    if (output.ToString().Contains(_prompt))
                    {
                        return output.ToString();
                    }
                    else if (!storeOutput && output.Length>_prompt.Length)
                    {
                        string tmp = output.ToString();
                        output.Clear();
                        output.Append(tmp.Substring(tmp.Length -  _prompt.Length));
                    }
                }

                Thread.Sleep(1000);
                
                if (!_client.IsConnected)
                {
                    throw new IOException("Disconnected");
                }
            }
        }

        public Script(string host = null, string sshConfigDir = null, params string[] aInitCommands)
        {
            _host = host;
            _sshConfigDir = sshConfigDir;
            initCommands.AddRange(aInitCommands);

            if (host != null)
            {
                connect();
                _reconnectLoop = Task.Run(() =>
                {
                    while (!cancelSrc.Token.IsCancellationRequested)
                    {
                        try
                        {
                            Task.Delay(30000, cancelSrc.Token).Wait();

                            if (!_client.IsConnected && !cancelSrc.Token.IsCancellationRequested)
                            {
                                connect();
                            }
                        }
                        catch (Exception ex) { }

                    }
                }, cancelSrc.Token);
            }
        }

        public bool connect()
        {
            ForwardedPortLocal jumpPort;

            string sshConfigFile = $@"{_sshConfigDir}\config";
            Regex hostsRex = new Regex(@"host\s(?<hostname>[a-z0-9]+)(.|\s)+?(?=(host\s)|$)");
            Match[] hostsMatches = hostsRex.Matches(File.ReadAllText(sshConfigFile)).ToArray();

            Match hostMatch = hostsMatches.First(hm => hm.Groups["hostname"].Value.Equals(_host));
            var hostConfig = new Regex(
                @"^\s*(?<key>\S+)\s+(?<value>.+?)\s*$",
                RegexOptions.Multiline);

            Match[] hostConfigAttr = hostConfig.Matches(hostMatch.Value).ToArray();

            string jumpHost = hostConfigAttr.First(hca => hca.Groups["key"].Value.Equals("ProxyJump", StringComparison.OrdinalIgnoreCase)).Groups["value"].Value;
            Match jumpHostMatch = hostsMatches.First(hm => hm.Groups["hostname"].Value.Equals(jumpHost));

            if (jumpHostMatch.Success)
            {
                Match[] jumpHostConfigAttr = hostConfig.Matches(jumpHostMatch.Value).ToArray();

                _jump = new SshClient(
                    jumpHostConfigAttr.First(hca => hca.Groups["key"].Value.Equals("HostName", StringComparison.OrdinalIgnoreCase)).Groups["value"].Value,
                    jumpHostConfigAttr.First(hca => hca.Groups["key"].Value.Equals("User", StringComparison.OrdinalIgnoreCase)).Groups["value"].Value,
                    new PrivateKeyFile(
                        @$"{_sshConfigDir}\" +
                        (jumpHostConfigAttr.First(hca => hca.Groups["key"].Value.Equals("IdentityFile", StringComparison.OrdinalIgnoreCase)).Groups["value"].Value).Split('/', '\\').Last()
                    )
                );

                _jump.Connect();

                jumpPort = new ForwardedPortLocal(
                    "127.0.0.1",
                    hostConfigAttr.First(hca => hca.Groups["key"].Value == "HostName").Groups["value"].Value,
                    22
                );

                _jump.AddForwardedPort(jumpPort);
                jumpPort.Start();

                _client = new SshClient(
               "127.0.0.1",
               (int)jumpPort.BoundPort,
               hostConfigAttr.First(hca => hca.Groups["key"].Value.Equals("User", StringComparison.OrdinalIgnoreCase)).Groups["value"].Value,
                   new PrivateKeyFile(
                       @$"{_sshConfigDir}\" +
                       (hostConfigAttr.First(hca => hca.Groups["key"].Value.Equals("IdentityFile", StringComparison.OrdinalIgnoreCase)).Groups["value"].Value).Split('/', '\\').Last()
                   )
                );
            }
            else
            {
                _client = new SshClient(
                    hostConfigAttr.First(hca => hca.Groups["key"].Value.Equals("HostName", StringComparison.OrdinalIgnoreCase)).Groups["value"].Value,
                    hostConfigAttr.First(hca => hca.Groups["key"].Value.Equals("User", StringComparison.OrdinalIgnoreCase)).Groups["value"].Value,
                    new PrivateKeyFile(
                        @$"{_sshConfigDir}\" +
                        (hostConfigAttr.First(hca => hca.Groups["key"].Value.Equals("IdentityFile", StringComparison.OrdinalIgnoreCase)).Groups["value"].Value).Split('/', '\\').Last()
                    )
                );
            }



            _client.Connect();

            _shell = _client.CreateShellStream(
                "xterm",
                120,
                40,
                800,
                600,
                4096
            );

            // Start login bash and give it a known prompt
            _shell.WriteLine("bash --login");
            _shell.WriteLine($"export PS1='{_prompt}'");
            ReadUntil();

            Script s = this;

            initCommands.ForEach(cmd => s >>= cmd);

            return true;
        }

        public static Script operator >>(Script left, string cmd)
        {
            bool success = false;

            if (left._shell != null)
            {               
                while (!success)
                {
                    try
                    {
                        Console.WriteLine(cmd);
                        left._shell.WriteLine(cmd);
                        left.LastOutput = left.ReadUntil();
                        success = true;
                        break;
                    }
                    catch (Exception e)
                    {

                    }

                    while (!left._client.IsConnected && !left.cancelSrc.IsCancellationRequested)
                    {
                        try
                        {
                            Task.Delay(30000, left.cancelSrc.Token).Wait();
                        }
                        catch (TaskCanceledException e)
                        {
                            throw e;
                        }
                        catch
                        {

                        }
                    }

                    Console.WriteLine($"cd {left._pwd}");
                    left._shell.WriteLine($"cd {left._pwd}");
                    left.ReadUntil(false);
                }

                // to keep track of current directory on remote
                if (cmd.StartsWith("cd "))
                {
                    left._pwd = cmd.Replace("cd ", "");
                }

                Console.WriteLine(left.LastOutput);
                return left;
            }

            return left;
        }

        public static Script operator |(Script left, string cmd)
        {
            bool success = false;

            if (left._shell != null)
            {
                while (!success)
                {
                    Console.WriteLine(cmd);
                    left._shell.WriteLine(cmd);

                    try
                    {
                        left.LastOutput = left.ReadUntil(false);
                        success = true;
                        break;
                    }
                    catch (IOException e)
                    {

                    }

                    while (!left._client.IsConnected && !left.cancelSrc.IsCancellationRequested)
                    {
                        try
                        {
                            Task.Delay(30000, left.cancelSrc.Token).Wait();
                        }
                        catch (TaskCanceledException e)
                        {
                            throw e;
                        }
                        catch
                        {

                        }
                    }

                    Console.WriteLine(cmd);
                    left._shell.WriteLine($"cd {left._pwd}");
                }

                // to keep track of current directory on remote
                if (cmd.StartsWith("cd "))
                {
                    left._pwd = left.LastOutput.Replace(_prompt, "");
                }

                Console.WriteLine(left.LastOutput);
                return left;
            }

            return left;
        }

        public static implicit operator string(Script script)
        {
            return script.LastOutput;
        }

        public bool Disconnect()
        {
            _isShutdown = true;

            if (_client!=null && _client.IsConnected)
            {
                cancelSrc.Cancel();
                try
                {
                    _reconnectLoop.Wait();
                }
                catch (Exception e)
                { }

                try
                {
                    _client.Disconnect();
                }
                catch
                { }
            }


            return !_client.IsConnected;
        }

    }
}
    
