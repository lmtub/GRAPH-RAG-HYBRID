static void check_pred16x16(H264PredContext *h, uint8_t *buf0, uint8_t *buf1,

                            int codec, int chroma_format, int bit_depth)

{

    if (chroma_format == 1) {

        int pred_mode;

        declare_func(void, uint8_t *src, ptrdiff_t stride);



        for (pred_mode = 0; pred_mode < 9; pred_mode++) {

            if (check_pred_func(h->pred16x16[pred_mode], "16x16", pred16x16_modes[codec][pred_mode])) {

                randomize_buffers();

                call_ref(src0, 48);

                call_new(src1, 48);

                if (memcmp(buf0, buf1, BUF_SIZE))

                    fail();

                bench_new(src1, 48);

            }

        }

    }

}
