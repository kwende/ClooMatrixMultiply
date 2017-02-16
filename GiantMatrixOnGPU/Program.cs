using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GiantMatrixOnGPU
{
    class Program
    {
        static void Main(string[] args)
        {
            const int MatrixWidth = 512*2;
            const int MatrixHeight = 424*2;
            const int NumberOfIterations = 10;

            float[] matrix1 = MatrixBuilder.BuildMatrix(MatrixWidth, MatrixHeight);
            float[] matrix2 = MatrixBuilder.BuildMatrix(MatrixWidth, MatrixHeight);

            CPUMatrixMultiplier cpu = new CPUMatrixMultiplier();
            GPUMatrixMultiplier gpu = new GPUMatrixMultiplier();

            Stopwatch cpuSw = new Stopwatch();
            cpuSw.Start();
            for (int c = 0; c < NumberOfIterations; c++)
            {
                float[] result = cpu.MultiplyMatrices(matrix1, matrix2, MatrixHeight, MatrixHeight, MatrixWidth);
            }
            cpuSw.Stop();

            Stopwatch gpuSw = new Stopwatch();
            gpuSw.Start();
            for (int c = 0; c < NumberOfIterations; c++)
            {
                float[] result = gpu.MultiplyMatrices(matrix1, matrix2, MatrixHeight, MatrixHeight, MatrixWidth);
            }
            gpuSw.Stop();

            Console.WriteLine($"CPU Matrix multiplication: {cpuSw.ElapsedMilliseconds / (NumberOfIterations * 1.0f)}ms");
            Console.WriteLine($"GPU Matrix multiplication: {gpuSw.ElapsedMilliseconds / (NumberOfIterations * 1.0f)}ms");
            Console.WriteLine($"GPU is {cpuSw.ElapsedMilliseconds / (gpuSw.ElapsedMilliseconds * 1.0f)}x faster."); 
            Console.ReadLine();

            return;
        }
    }
}
