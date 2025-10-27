static void check_pred8x8(H264PredContext *h, uint8_t *buf0, uint8_t *buf1,

                          int codec, int chroma_format, int bit_depth)

{

    int pred_mode;

    for (pred_mode = 0; pred_mode < 11; pred_mode++) {

        if (check_pred_func(h->pred8x8[pred_mode], (chroma_format == 2) ? "8x16" : "8x8",

                            pred8x8_modes[codec][pred_mode])) {

            randomize_buffers();

            call_ref(src0, (ptrdiff_t)24*SIZEOF_PIXEL);

            call_new(src1, (ptrdiff_t)24*SIZEOF_PIXEL);

            if (memcmp(buf0, buf1, BUF_SIZE))

                fail();

            bench_new(src1, (ptrdiff_t)24*SIZEOF_PIXEL);

        }

    }

}
