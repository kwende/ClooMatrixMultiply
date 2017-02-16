using Cloo;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GiantMatrixOnGPU
{
    public class GPUMatrixMultiplier
    {
        private bool _initialized = false;
        private ComputePlatform _integratedIntelGPUPlatform;
        private ComputeContext _context;
        private ComputeCommandQueue _commandQueue;
        private ComputeProgram _program;
        private ComputeKernel _kernel;


        public float[] MultiplyMatricesZeroCopy(float[] matrix1, float[] matrix2,
            int matrix1Height, int matrix1WidthMatrix2Height, int matrix2Width)
        {
            if (!_initialized)
            {
                Initialize();
                _initialized = true;
            }

            ComputeBuffer<float> matrix1Buffer = new ComputeBuffer<float>(_context,
                ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer,
                matrix1);
            _kernel.SetMemoryArgument(0, matrix1Buffer);

            ComputeBuffer<float> matrix2Buffer = new ComputeBuffer<float>(_context,
                ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer,
                matrix2);
            _kernel.SetMemoryArgument(1, matrix2Buffer);

            float[] ret = new float[matrix1Height * matrix2Width];
            ComputeBuffer<float> retBuffer = new ComputeBuffer<float>(_context,
                ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.UseHostPointer,
                ret);
            _kernel.SetMemoryArgument(2, retBuffer);

            _kernel.SetValueArgument<int>(3, matrix1WidthMatrix2Height);
            _kernel.SetValueArgument<int>(4, matrix2Width);

            _commandQueue.Execute(_kernel,
                new long[] { 0 },
                new long[] { matrix2Width, matrix1Height },
                null, null);

            IntPtr retPtr = _commandQueue.Map(
                retBuffer,
                true,
                ComputeMemoryMappingFlags.Read,
                0,
                ret.Length, null);

            _commandQueue.Unmap(retBuffer, ref retPtr, null);
            //_commandQueue.Finish();

            matrix1Buffer.Dispose();
            matrix2Buffer.Dispose();
            retBuffer.Dispose();

            return ret;
        }

        public float[] MultiplyMatrices(float[] matrix1, float[] matrix2,
            int matrix1Height, int matrix1WidthMatrix2Height, int matrix2Width)
        {
            if (!_initialized)
            {
                Initialize();
                _initialized = true;
            }

            ComputeBuffer<float> matrix1Buffer = new ComputeBuffer<float>(_context,
                ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer,
                matrix1);
            _kernel.SetMemoryArgument(0, matrix1Buffer);

            ComputeBuffer<float> matrix2Buffer = new ComputeBuffer<float>(_context,
                ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer,
                matrix2);
            _kernel.SetMemoryArgument(1, matrix2Buffer);

            float[] ret = new float[matrix1Height * matrix2Width];
            ComputeBuffer<float> retBuffer = new ComputeBuffer<float>(_context,
                ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer,
                ret);
            _kernel.SetMemoryArgument(2, retBuffer);

            _kernel.SetValueArgument<int>(3, matrix1WidthMatrix2Height);
            _kernel.SetValueArgument<int>(4, matrix2Width);

            _commandQueue.Execute(_kernel,
                new long[] { 0 },
                new long[] { matrix2Width, matrix1Height },
                null, null);

            unsafe
            {
                fixed (float* retPtr = ret)
                {
                    _commandQueue.Read(retBuffer,
                        false, 0,
                        ret.Length,
                        new IntPtr(retPtr),
                        null);

                    _commandQueue.Finish();
                }
            }

            matrix1Buffer.Dispose();
            matrix2Buffer.Dispose();
            retBuffer.Dispose();

            return ret;
        }

        private void Initialize()
        {
            // get the intel integrated GPU
            _integratedIntelGPUPlatform = ComputePlatform.Platforms.Where(n => n.Name.Contains("Intel")).First();

            // create the compute context. 
            _context = new ComputeContext(
                ComputeDeviceTypes.Gpu, // use the gpu
                new ComputeContextPropertyList(_integratedIntelGPUPlatform), // use the intel openCL platform
                null,
                IntPtr.Zero);

            // the command queue is the, well, queue of commands sent to the "device" (GPU)
            _commandQueue = new ComputeCommandQueue(
                _context, // the compute context
                _context.Devices[0], // first device matching the context specifications
                ComputeCommandQueueFlags.None); // no special flags

            string kernelSource = null;
            using (StreamReader sr = new StreamReader("kernel.cl"))
            {
                kernelSource = sr.ReadToEnd();
            }

            // create the "program"
            _program = new ComputeProgram(_context, new string[] { kernelSource });

            // compile. 
            _program.Build(null, null, null, IntPtr.Zero);
            _kernel = _program.CreateKernel("ComputeMatrix");
        }
    }
}
