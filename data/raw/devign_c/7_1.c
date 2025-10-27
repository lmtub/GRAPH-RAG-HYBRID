static void check_lowpass_line(int depth){

    LOCAL_ALIGNED_32(uint8_t, src,     [SRC_SIZE]);

    LOCAL_ALIGNED_32(uint8_t, dst_ref, [WIDTH_PADDED]);

    LOCAL_ALIGNED_32(uint8_t, dst_new, [WIDTH_PADDED]);

    int w = WIDTH;

    int mref = WIDTH_PADDED * -1;

    int pref = WIDTH_PADDED;

    int i, depth_byte;

    InterlaceContext s;



    declare_func(void, uint8_t *dstp, ptrdiff_t linesize, const uint8_t *srcp,

                 ptrdiff_t mref, ptrdiff_t pref, int clip_max);



    s.lowpass = 1;

    s.lowpass = VLPF_LIN;

    depth_byte = depth >> 3;

    w /= depth_byte;



    memset(src,     0, SRC_SIZE);

    memset(dst_ref, 0, WIDTH_PADDED);

    memset(dst_new, 0, WIDTH_PADDED);

    randomize_buffers(src, SRC_SIZE);



    ff_interlace_init(&s, depth);



    if (check_func(s.lowpass_line, "lowpass_line_%d", depth)) {

        for (i = 0; i < 32; i++) { /* simulate crop */

            call_ref(dst_ref, w, src + WIDTH_PADDED, mref - i*depth_byte, pref, 0);

            call_new(dst_new, w, src + WIDTH_PADDED, mref - i*depth_byte, pref, 0);

            if (memcmp(dst_ref, dst_new, WIDTH - i))

                fail();

        }

        bench_new(dst_new, w, src + WIDTH_PADDED, mref, pref, 0);

    }

}
