static void predictor_decode_mono(APEContext *ctx, int count)

{

    APEPredictor *p = &ctx->predictor;

    int32_t *decoded0 = ctx->decoded[0];

    int32_t predictionA, currentA, A, sign;



    currentA = p->lastA[0];



    while (count--) {

        A = *decoded0;



        p->buf[YDELAYA] = currentA;

        p->buf[YDELAYA - 1] = p->buf[YDELAYA] - p->buf[YDELAYA - 1];



        predictionA = p->buf[YDELAYA    ] * p->coeffsA[0][0] +

                      p->buf[YDELAYA - 1] * p->coeffsA[0][1] +

                      p->buf[YDELAYA - 2] * p->coeffsA[0][2] +

                      p->buf[YDELAYA - 3] * p->coeffsA[0][3];



        currentA = A + (predictionA >> 10);



        p->buf[YADAPTCOEFFSA]     = APESIGN(p->buf[YDELAYA    ]);

        p->buf[YADAPTCOEFFSA - 1] = APESIGN(p->buf[YDELAYA - 1]);



        sign = APESIGN(A);

        p->coeffsA[0][0] += p->buf[YADAPTCOEFFSA    ] * sign;

        p->coeffsA[0][1] += p->buf[YADAPTCOEFFSA - 1] * sign;

        p->coeffsA[0][2] += p->buf[YADAPTCOEFFSA - 2] * sign;

        p->coeffsA[0][3] += p->buf[YADAPTCOEFFSA - 3] * sign;



        p->buf++;



        /* Have we filled the history buffer? */

        if (p->buf == p->historybuffer + HISTORY_SIZE) {

            memmove(p->historybuffer, p->buf,

                    PREDICTOR_SIZE * sizeof(*p->historybuffer));

            p->buf = p->historybuffer;

        }



        p->filterA[0] = currentA + ((p->filterA[0] * 31) >> 5);

        *(decoded0++) = p->filterA[0];

    }



    p->lastA[0] = currentA;

}
