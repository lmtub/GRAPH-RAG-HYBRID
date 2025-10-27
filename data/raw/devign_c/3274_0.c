static float ssim_plane(uint8_t *main, int main_stride,

                        uint8_t *ref, int ref_stride,

                        int width, int height, void *temp)

{

    int z = 0;

    int x, y;

    float ssim = 0.0;

    int (*sum0)[4] = temp;

    int (*sum1)[4] = sum0 + (width >> 2) + 3;



    width >>= 2;

    height >>= 2;



    for (y = 1; y < height; y++) {

        for (; z <= y; z++) {

            FFSWAP(void*, sum0, sum1);

            for (x = 0; x < width; x+=2)

                ssim_4x4x2_core(&main[4 * (x + z * main_stride)], main_stride,

                                &ref[4 * (x + z * ref_stride)], ref_stride,

                                &sum0[x]);

        }



        ssim += ssim_endn(sum0, sum1, width - 1);

    }



    return ssim / ((height - 1) * (width - 1));

}
