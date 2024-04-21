#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <math.h>

#define WIDTH 8192
#define HEIGTH WIDTH
#define BLOCK_WIDTH 32
#define BLOCK_HEIGTH BLOCK_WIDTH

#define MAX_ITER 1000
#define MLX -1.235
#define MHX 1.235
#define MMX -0.765
#define MLY -1.12
#define MHY 1.12
#define MMY 0
#define ZOOM 1

struct color {
    unsigned char r;
    unsigned char g;
    unsigned char b;
};
typedef struct color color;

__global__ void kernel(color* image, int w, int h) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    float u = (float)col / (float)w;
    float v = (float)row / (float)h;
    if (col < w && row < h) {
        /*
        if (pow(u-0.5, 2) + pow(v-0.5, 2) < 0.25) {
            image[col + row * w].r = UCHAR_MAX * u;
            image[col + row * w].g = UCHAR_MAX * v;
            image[col + row * w].b = 127;
        }
        */
        float a = (MHX) / (ZOOM*ZOOM) + MMX;
        float b = (MLX) / (ZOOM*ZOOM) + MMX;
        float x0 = (a - b) * u + b;
        float c = (MHY) / (ZOOM*ZOOM) + MMY;
        float d = (MLY) / (ZOOM*ZOOM) + MMY;
        float y0 = (c - d) * v + d;
        float x = 0;
        float y = 0;
        int iter;
        for (iter = 0; x * x - y * y <= 2 * 2 && iter < MAX_ITER; iter++) {
            float xtmp = x * x - y * y + x0;
            y = 2 * x * y + y0;
            x = xtmp;
        }
        //unsigned char brightness = float(UCHAR_MAX) * (1 - float(iter) / float(MAX_ITER));
        //image[col + row * w] = { brightness, brightness, brightness };
        image[col + row * w] = { 
            unsigned char((iter * 3) % UCHAR_MAX), 
            unsigned char((iter * 4) % UCHAR_MAX), 
            unsigned char((iter * 5) % UCHAR_MAX)
        };
    }
}

class Image {
private:
    color* host;
    color* device;
    int height;
    int width;
    cudaError_t cerr;
public:
    Image(int x, int y) {
        width = x;
        height = y;
        cerr = cudaMalloc((void**)&device, x * y * sizeof(color));
        assert(!cerr);
        host = new color[x * y]();
    }
    void render(const std::string& filename) {
        dim3 grid(width/BLOCK_WIDTH, height/BLOCK_HEIGTH);
        dim3 block(BLOCK_WIDTH, BLOCK_HEIGTH);
        clock_t tic = clock();
        kernel << <grid, block >> > (device, width, height);
        cudaDeviceSynchronize();
        clock_t toc = clock();
        printf("kernel: %f\n", double(toc - tic) / CLOCKS_PER_SEC);
        tic = clock();
        cerr = cudaMemcpy(host, device, width * height * sizeof(color), cudaMemcpyDeviceToHost);
        toc = clock();
        assert(!cerr);
        printf("copy: %f\n", double(toc - tic) / CLOCKS_PER_SEC);

        tic = clock();
        std::ofstream ppmFile(filename, std::ios::out | std::ios::binary);
		if (!ppmFile) {
			std::cerr << "Error opening file: " << filename << std::endl;
			return;
		}

		ppmFile << "P6\n";
		ppmFile << width << " " << height << "\n";
		ppmFile << UCHAR_MAX << "\n";
        ppmFile.write((char*)host, width * height * sizeof(color));

		ppmFile.close();
        toc = clock();
        printf("file write: %f\n", double(toc - tic) / CLOCKS_PER_SEC);
    }
    ~Image() {
        cudaFree(device);
        delete[] host;
    }
};

int main()
{
    Image image = Image(WIDTH, HEIGTH);
    image.render("output.ppm");
    return 0;
}

/*
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
*/