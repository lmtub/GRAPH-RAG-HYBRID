void rgb24toyv12_c(const uint8_t *src, uint8_t *ydst, uint8_t *udst,

                   uint8_t *vdst, int width, int height, int lumStride,

                   int chromStride, int srcStride)

{

    int y;

    const int chromWidth = width >> 1;



    for (y = 0; y < height; y += 2) {

        int i;

        for (i = 0; i < chromWidth; i++) {

            unsigned int b = src[6 * i + 0];

            unsigned int g = src[6 * i + 1];

            unsigned int r = src[6 * i + 2];



            unsigned int Y = ((RY * r + GY * g + BY * b) >> RGB2YUV_SHIFT) +  16;

            unsigned int V = ((RV * r + GV * g + BV * b) >> RGB2YUV_SHIFT) + 128;

            unsigned int U = ((RU * r + GU * g + BU * b) >> RGB2YUV_SHIFT) + 128;



            udst[i]     = U;

            vdst[i]     = V;

            ydst[2 * i] = Y;



            b = src[6 * i + 3];

            g = src[6 * i + 4];

            r = src[6 * i + 5];



            Y = ((RY * r + GY * g + BY * b) >> RGB2YUV_SHIFT) + 16;

            ydst[2 * i + 1] = Y;

        }

        ydst += lumStride;

        src  += srcStride;



        if (y+1 == height)

            break;



        for (i = 0; i < chromWidth; i++) {

            unsigned int b = src[6 * i + 0];

            unsigned int g = src[6 * i + 1];

            unsigned int r = src[6 * i + 2];



            unsigned int Y = ((RY * r + GY * g + BY * b) >> RGB2YUV_SHIFT) + 16;



            ydst[2 * i] = Y;



            b = src[6 * i + 3];

            g = src[6 * i + 4];

            r = src[6 * i + 5];



            Y = ((RY * r + GY * g + BY * b) >> RGB2YUV_SHIFT) + 16;

            ydst[2 * i + 1] = Y;

        }

        udst += chromStride;

        vdst += chromStride;

        ydst += lumStride;

        src  += srcStride;

    }

}
