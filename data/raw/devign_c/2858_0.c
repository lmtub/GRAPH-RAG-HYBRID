static void check_pred4x4(H264PredContext *h, uint8_t *buf0, uint8_t *buf1,

                          int codec, int chroma_format, int bit_depth)

{

    if (chroma_format == 1) {

        uint8_t *topright = buf0 + 2*16;

        int pred_mode;

        for (pred_mode = 0; pred_mode < 15; pred_mode++) {

            if (check_pred_func(h->pred4x4[pred_mode], "4x4", pred4x4_modes[codec][pred_mode])) {

                randomize_buffers();

                call_ref(src0, topright, (ptrdiff_t)12*SIZEOF_PIXEL);

                call_new(src1, topright, (ptrdiff_t)12*SIZEOF_PIXEL);

                if (memcmp(buf0, buf1, BUF_SIZE))

                    fail();

                bench_new(src1, topright, (ptrdiff_t)12*SIZEOF_PIXEL);

            }

        }

    }

}
