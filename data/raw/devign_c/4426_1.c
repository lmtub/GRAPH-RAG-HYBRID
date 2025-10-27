static int dx2_decode_slice_444(GetBitContext *gb, AVFrame *frame,

                                int line, int left,

                                uint8_t lru[3][8])

{

    int x, y;



    int width   = frame->width;



    int ystride = frame->linesize[0];

    int ustride = frame->linesize[1];

    int vstride = frame->linesize[2];



    uint8_t *Y  = frame->data[0] + ystride * line;

    uint8_t *U  = frame->data[1] + ustride * line;

    uint8_t *V  = frame->data[2] + vstride * line;



    for (y = 0; y < left && get_bits_left(gb) > 16; y++) {

        for (x = 0; x < width; x++) {

            Y[x] = decode_sym(gb, lru[0]);

            U[x] = decode_sym(gb, lru[1]) ^ 0x80;

            V[x] = decode_sym(gb, lru[2]) ^ 0x80;

        }



        Y += ystride;

        U += ustride;

        V += vstride;

    }



    return y;

}
