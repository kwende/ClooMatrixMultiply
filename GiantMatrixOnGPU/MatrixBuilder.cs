using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GiantMatrixOnGPU
{
    public static class MatrixBuilder
    {
        public static float[] BuildMatrix(int width, int height)
        {
            Random rand = new Random(1234);
            float[] matrix = new float[width * height]; 
            for(int i=0;i<width*height;i++)
            {
                matrix[i] = (float)rand.NextDouble(); 
            }
            return matrix; 
        }
    }
}
