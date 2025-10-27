static void check_threshold_8(void){

    LOCAL_ALIGNED_32(uint8_t, in       , [WIDTH_PADDED]);

    LOCAL_ALIGNED_32(uint8_t, threshold, [WIDTH_PADDED]);

    LOCAL_ALIGNED_32(uint8_t, min      , [WIDTH_PADDED]);

    LOCAL_ALIGNED_32(uint8_t, max      , [WIDTH_PADDED]);

    LOCAL_ALIGNED_32(uint8_t, out_ref  , [WIDTH_PADDED]);

    LOCAL_ALIGNED_32(uint8_t, out_new  , [WIDTH_PADDED]);

    ptrdiff_t line_size = WIDTH_PADDED;

    int w = WIDTH;



    ThresholdContext s;

    s.depth = 8;

    ff_threshold_init(&s);



    declare_func(void, const uint8_t *in, const uint8_t *threshold,

                 const uint8_t *min, const uint8_t *max, uint8_t *out,

                 ptrdiff_t ilinesize, ptrdiff_t tlinesize,

                 ptrdiff_t flinesize, ptrdiff_t slinesize,

                 ptrdiff_t olinesize, int w, int h);



    memset(in,     0, WIDTH_PADDED);

    memset(threshold, 0, WIDTH_PADDED);

    memset(min, 0, WIDTH_PADDED);

    memset(max, 0, WIDTH_PADDED);

    memset(out_ref, 0, WIDTH_PADDED);

    memset(out_new, 0, WIDTH_PADDED);

    randomize_buffers(in, WIDTH);

    randomize_buffers(threshold, WIDTH);

    randomize_buffers(min, WIDTH);

    randomize_buffers(max, WIDTH);



    if (check_func(s.threshold, "threshold8")) {

        call_ref(in, threshold, min, max, out_ref, line_size, line_size, line_size, line_size, line_size, w, 1);

        call_new(in, threshold, min, max, out_new, line_size, line_size, line_size, line_size, line_size, w, 1);

        if (memcmp(out_ref, out_new, w))

            fail();

        bench_new(in, threshold, min, max, out_new, line_size, line_size, line_size, line_size, line_size, w, 1);

    }

}
