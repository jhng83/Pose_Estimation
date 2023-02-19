﻿using System;
using System.Diagnostics;

namespace RunPythonScript
{
    class Program
    {
        static void Main(string[] args)
        {
            // specify the path to the Anaconda environment activation script
            string activateScript = @"C:\Users\JHmachine\anaconda3\Scripts\activate.bat";

            // specify the name of the Anaconda environment
            string environmentName = "py38";

            // specify the path to the Python script that you want to run
            string scriptPath = @"C:\Users\JHmachine\Desktop\app\pose.py";

            // specify the command to activate the Anaconda environment and run the Python script
            string command = $@"call {activateScript} {environmentName} && python {scriptPath}";

            // start a new process to run the command
            Process process = new Process();
            process.StartInfo.FileName = "cmd.exe";
            process.StartInfo.Arguments = $@"/c ""{command}""";
            process.StartInfo.UseShellExecute = false;
            process.StartInfo.RedirectStandardOutput = true;
            process.Start();
            process.WaitForExit();

            // print the output of the Python script to the console
            Console.WriteLine(process.StandardOutput.ReadToEnd());
        }
    }
}