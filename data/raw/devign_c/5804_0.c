static void check_pred8x8l(H264PredContext *h, uint8_t *buf0, uint8_t *buf1,

                           int codec, int chroma_format, int bit_depth)

{

    if (chroma_format == 1 && codec_ids[codec] == AV_CODEC_ID_H264) {

        int pred_mode;

        for (pred_mode = 0; pred_mode < 12; pred_mode++) {

            if (check_pred_func(h->pred8x8l[pred_mode], "8x8l", pred4x4_modes[codec][pred_mode])) {

                int neighbors;

                for (neighbors = 0; neighbors <= 0xc000; neighbors += 0x4000) {

                    int has_topleft  = neighbors & 0x8000;

                    int has_topright = neighbors & 0x4000;



                    if ((pred_mode == DIAG_DOWN_RIGHT_PRED || pred_mode == VERT_RIGHT_PRED) && !has_topleft)

                        continue; /* Those aren't allowed according to the spec */



                    randomize_buffers();

                    call_ref(src0, has_topleft, has_topright, (ptrdiff_t)24*SIZEOF_PIXEL);

                    call_new(src1, has_topleft, has_topright, (ptrdiff_t)24*SIZEOF_PIXEL);

                    if (memcmp(buf0, buf1, BUF_SIZE))

                        fail();

                    bench_new(src1, has_topleft, has_topright, (ptrdiff_t)24*SIZEOF_PIXEL);

                }

            }

        }

    }

}
