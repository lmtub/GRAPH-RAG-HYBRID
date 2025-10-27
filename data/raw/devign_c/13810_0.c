static void do_rematrixing(AC3DecodeContext *ctx)

{

    ac3_audio_block *ab = &ctx->audio_block;

    uint8_t bnd1 = 13, bnd2 = 25, bnd3 = 37, bnd4 = 61;

    uint8_t bndend;



    bndend = FFMIN(ab->endmant[0], ab->endmant[1]);

    if (ab->rematflg & 1)

        _do_rematrixing(ctx, bnd1, bnd2);

    if (ab->rematflg & 2)

        _do_rematrixing(ctx, bnd2, bnd3);

    if (ab->rematflg & 4) {

        if (ab->cplbegf > 0 && ab->cplbegf <= 2 && (ab->flags & AC3_AB_CPLINU))

            _do_rematrixing(ctx, bnd3, bndend);

        else {

            _do_rematrixing(ctx, bnd3, bnd4);

            if (ab->rematflg & 8)

                _do_rematrixing(ctx, bnd4, bndend);

        }

    }

}
