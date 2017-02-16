kernel void ComputeMatrix(
    global read_only float* matrix1,
    global read_only float* matrix2,
    global write_only float* output, 
    int matrix1WidthMatrix2Height,
    int matrix2Width)
{
    int x = get_global_id(0); 
    int y = get_global_id(1); 
    int i = y * matrix2Width + x; 

    float value = 0.0f; 
    // row y of matrix1 * column x of matrix2
    for (int c = 0; c < matrix1WidthMatrix2Height; c++)
    {
        int m1Index = y * matrix1WidthMatrix2Height + c;
        int m2Index = c * matrix2Width + x;

        value += matrix1[m1Index] * matrix2[m2Index]; 
    }
    output[i] = value; 
}
