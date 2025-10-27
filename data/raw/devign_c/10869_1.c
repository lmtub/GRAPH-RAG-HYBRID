static void check_add_res(HEVCDSPContext h, int bit_depth)

{

    int i;

    LOCAL_ALIGNED_32(int16_t, res0, [32 * 32]);

    LOCAL_ALIGNED_32(int16_t, res1, [32 * 32]);

    LOCAL_ALIGNED_32(uint8_t, dst0, [32 * 32 * 2]);

    LOCAL_ALIGNED_32(uint8_t, dst1, [32 * 32 * 2]);



    for (i = 2; i <= 5; i++) {

        int block_size = 1 << i;

        int size = block_size * block_size;

        ptrdiff_t stride = block_size << (bit_depth > 8);

        declare_func_emms(AV_CPU_FLAG_MMX, void, uint8_t *dst, int16_t *res, ptrdiff_t stride);



        randomize_buffers(res0, size);

        randomize_buffers2(dst0, size);

        memcpy(res1, res0, sizeof(*res0) * size);

        memcpy(dst1, dst0, size);



        if (check_func(h.add_residual[i - 2], "add_res_%dx%d_%d", block_size, block_size, bit_depth)) {

            call_ref(dst0, res0, stride);

            call_new(dst1, res1, stride);

            if (memcmp(dst0, dst1, size))

                fail();

            bench_new(dst1, res1, stride);

        }

    }

}
