static void check_loopfilter(void)

{

    LOCAL_ALIGNED_32(uint8_t, base0, [32 + 16 * 16 * 2]);

    LOCAL_ALIGNED_32(uint8_t, base1, [32 + 16 * 16 * 2]);

    VP9DSPContext dsp;

    int dir, wd, wd2, bit_depth;

    static const char *const dir_name[2] = { "h", "v" };

    static const int E[2] = { 20, 28 }, I[2] = { 10, 16 };

    static const int H[2] = { 7, 11 }, F[2] = { 1, 1 };

    declare_func(void, uint8_t *dst, ptrdiff_t stride, int E, int I, int H);



    for (bit_depth = 8; bit_depth <= 12; bit_depth += 2) {

        ff_vp9dsp_init(&dsp, bit_depth, 0);



        for (dir = 0; dir < 2; dir++) {

            int midoff = (dir ? 8 * 8 : 8) * SIZEOF_PIXEL;

            int midoff_aligned = (dir ? 8 * 8 : 16) * SIZEOF_PIXEL;

            uint8_t *buf0 = base0 + midoff_aligned;

            uint8_t *buf1 = base1 + midoff_aligned;



            for (wd = 0; wd < 3; wd++) {

                // 4/8/16wd_8px

                if (check_func(dsp.loop_filter_8[wd][dir],

                               "vp9_loop_filter_%s_%d_8_%dbpp",

                               dir_name[dir], 4 << wd, bit_depth)) {

                    randomize_buffers(0, 0, 8);

                    memcpy(buf1 - midoff, buf0 - midoff,

                           16 * 8 * SIZEOF_PIXEL);

                    call_ref(buf0, 16 * SIZEOF_PIXEL >> dir, E[0], I[0], H[0]);

                    call_new(buf1, 16 * SIZEOF_PIXEL >> dir, E[0], I[0], H[0]);

                    if (memcmp(buf0 - midoff, buf1 - midoff, 16 * 8 * SIZEOF_PIXEL))

                        fail();

                    bench_new(buf1, 16 * SIZEOF_PIXEL >> dir, E[0], I[0], H[0]);

                }

            }



            midoff = (dir ? 16 * 8 : 8) * SIZEOF_PIXEL;

            midoff_aligned = (dir ? 16 * 8 : 16) * SIZEOF_PIXEL;



            buf0 = base0 + midoff_aligned;

            buf1 = base1 + midoff_aligned;



            // 16wd_16px loopfilter

            if (check_func(dsp.loop_filter_16[dir],

                           "vp9_loop_filter_%s_16_16_%dbpp",

                           dir_name[dir], bit_depth)) {

                randomize_buffers(0, 0, 16);

                randomize_buffers(0, 8, 16);

                memcpy(buf1 - midoff, buf0 - midoff, 16 * 16 * SIZEOF_PIXEL);

                call_ref(buf0, 16 * SIZEOF_PIXEL, E[0], I[0], H[0]);

                call_new(buf1, 16 * SIZEOF_PIXEL, E[0], I[0], H[0]);

                if (memcmp(buf0 - midoff, buf1 - midoff, 16 * 16 * SIZEOF_PIXEL))

                    fail();

                bench_new(buf1, 16 * SIZEOF_PIXEL, E[0], I[0], H[0]);

            }



            for (wd = 0; wd < 2; wd++) {

                for (wd2 = 0; wd2 < 2; wd2++) {

                    // mix2 loopfilter

                    if (check_func(dsp.loop_filter_mix2[wd][wd2][dir],

                                   "vp9_loop_filter_mix2_%s_%d%d_16_%dbpp",

                                   dir_name[dir], 4 << wd, 4 << wd2, bit_depth)) {

                        randomize_buffers(0, 0, 16);

                        randomize_buffers(1, 8, 16);

                        memcpy(buf1 - midoff, buf0 - midoff, 16 * 16 * SIZEOF_PIXEL);

#define M(a) (((a)[1] << 8) | (a)[0])

                        call_ref(buf0, 16 * SIZEOF_PIXEL, M(E), M(I), M(H));

                        call_new(buf1, 16 * SIZEOF_PIXEL, M(E), M(I), M(H));

                        if (memcmp(buf0 - midoff, buf1 - midoff, 16 * 16 * SIZEOF_PIXEL))

                            fail();

                        bench_new(buf1, 16 * SIZEOF_PIXEL, M(E), M(I), M(H));

#undef M

                    }

                }

            }

        }

    }

    report("loopfilter");

}
