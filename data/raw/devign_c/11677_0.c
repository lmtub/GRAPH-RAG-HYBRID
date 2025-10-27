static void predictor_decode_stereo(APEContext *ctx, int count)

{

    APEPredictor *p = &ctx->predictor;

    int32_t *decoded0 = ctx->decoded[0];

    int32_t *decoded1 = ctx->decoded[1];



    while (count--) {

        /* Predictor Y */

        *decoded0 = predictor_update_filter(p, *decoded0, 0, YDELAYA, YDELAYB,

                                            YADAPTCOEFFSA, YADAPTCOEFFSB);

        decoded0++;

        *decoded1 = predictor_update_filter(p, *decoded1, 1, XDELAYA, XDELAYB,

                                            XADAPTCOEFFSA, XADAPTCOEFFSB);

        decoded1++;



        /* Combined */

        p->buf++;



        /* Have we filled the history buffer? */

        if (p->buf == p->historybuffer + HISTORY_SIZE) {

            memmove(p->historybuffer, p->buf,

                    PREDICTOR_SIZE * sizeof(*p->historybuffer));

            p->buf = p->historybuffer;

        }

    }

}
