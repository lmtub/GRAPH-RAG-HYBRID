static int dx2_decode_slice_420(GetBitContext *gb, AVFrame *frame,

                                int line, int left,

                                uint8_t lru[3][8])

{

    int x, y;



    int width    = frame->width;



    int ystride = frame->linesize[0];

    int ustride = frame->linesize[1];

    int vstride = frame->linesize[2];



    uint8_t *Y  = frame->data[0] + ystride * line;

    uint8_t *U  = frame->data[1] + (ustride >> 1) * line;

    uint8_t *V  = frame->data[2] + (vstride >> 1) * line;





    for (y = 0; y < left - 1 && get_bits_left(gb) > 16; y += 2) {

        for (x = 0; x < width; x += 2) {

            Y[x + 0 + 0 * ystride] = decode_sym(gb, lru[0]);

            Y[x + 1 + 0 * ystride] = decode_sym(gb, lru[0]);

            Y[x + 0 + 1 * ystride] = decode_sym(gb, lru[0]);

            Y[x + 1 + 1 * ystride] = decode_sym(gb, lru[0]);

            U[x >> 1] = decode_sym(gb, lru[1]) ^ 0x80;

            V[x >> 1] = decode_sym(gb, lru[2]) ^ 0x80;

        }



        Y += ystride << 1;

        U += ustride;

        V += vstride;

    }



    return y;

}
