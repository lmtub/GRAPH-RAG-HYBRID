static void do_bit_allocation(AC3DecodeContext *ctx, int flags)

{

    ac3_audio_block *ab = &ctx->audio_block;

    int i, snroffst = 0;



    if (!flags) /* bit allocation is not required */

        return;



    if (ab->flags & AC3_AB_SNROFFSTE) { /* check whether snroffsts are zero */

        snroffst += ab->csnroffst;

        if (ab->flags & AC3_AB_CPLINU)

            snroffst += ab->cplfsnroffst;

        for (i = 0; i < ctx->bsi.nfchans; i++)

            snroffst += ab->fsnroffst[i];

        if (ctx->bsi.flags & AC3_BSI_LFEON)

            snroffst += ab->lfefsnroffst;

        if (!snroffst) {

            memset(ab->cplbap, 0, sizeof (ab->cplbap));

            for (i = 0; i < ctx->bsi.nfchans; i++)

                memset(ab->bap[i], 0, sizeof (ab->bap[i]));

            memset(ab->lfebap, 0, sizeof (ab->lfebap));



            return;

        }

    }



    /* perform bit allocation */

    if ((ab->flags & AC3_AB_CPLINU) && (flags & 64))

        do_bit_allocation1(ctx, 5);

    for (i = 0; i < ctx->bsi.nfchans; i++)

        if (flags & (1 << i))

            do_bit_allocation1(ctx, i);

    if ((ctx->bsi.flags & AC3_BSI_LFEON) && (flags & 32))

        do_bit_allocation1(ctx, 6);

}
