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

            Stopwatch gpuSwCopy = new Stopwatch();
            gpuSwCopy.Start();
            for (int c = 0; c < NumberOfIterations; c++)
            {
                float[] result = gpu.MultiplyMatrices(matrix1, matrix2, MatrixHeight, MatrixHeight, MatrixWidth);
            }
            gpuSwCopy.Stop();

            Stopwatch gpuSwNoCopy = new Stopwatch();
            gpuSwNoCopy.Start();
            for (int c = 0; c < NumberOfIterations; c++)
            {
                float[] result = gpu.MultiplyMatricesZeroCopy(matrix1, matrix2, MatrixHeight, MatrixHeight, MatrixWidth);
            }
            gpuSwNoCopy.Stop();

            Console.WriteLine($"CPU Matrix multiplication: {cpuSw.ElapsedMilliseconds / (NumberOfIterations * 1.0f)}ms");
            Console.WriteLine($"GPU Matrix multiplication (copy): {gpuSwCopy.ElapsedMilliseconds / (NumberOfIterations * 1.0f)}ms");
            Console.WriteLine($"GPU Matrix multiplication (zero copy): {gpuSwNoCopy.ElapsedMilliseconds / (NumberOfIterations * 1.0f)}ms");
            Console.WriteLine($"GPU (w/ copy) is {cpuSw.ElapsedMilliseconds / (gpuSwCopy.ElapsedMilliseconds * 1.0f)}x faster.");
            Console.WriteLine($"GPU (zero copy) is {cpuSw.ElapsedMilliseconds / (gpuSwNoCopy.ElapsedMilliseconds * 1.0f)}x faster.");
            Console.ReadLine();

            return;
        }
    }
}
