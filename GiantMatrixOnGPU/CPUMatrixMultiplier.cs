using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GiantMatrixOnGPU
{
    public class CPUMatrixMultiplier
    {
        public float[] MultiplyMatrices(float[] matrix1, float[] matrix2,
            int matrix1Height, int matrix1WidthMatrix2Height, int matrix2Width)
        {
            float[] ret = new float[matrix1Height * matrix2Width];

            for (int y = 0, i = 0; y < matrix1Height; y++)
            {
                for (int x = 0; x < matrix2Width; x++, i++)
                {
                    float value = 0.0f; 
                    // row y of matrix1 * column x of matrix2
                    for (int c = 0; c < matrix1WidthMatrix2Height; c++)
                    {
                        int m1Index = y * matrix1WidthMatrix2Height + c;
                        int m2Index = c * matrix2Width + x;

                        value += matrix1[m1Index] * matrix2[m2Index]; 
                    }
                    ret[i] = value; 
                }
            }

            return ret;
        }
    }
}
