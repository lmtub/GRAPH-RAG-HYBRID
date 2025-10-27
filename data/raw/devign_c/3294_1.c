static int convert_zp2tf(AVFilterContext *ctx, int channels)

{

    AudioIIRContext *s = ctx->priv;

    int ch, i, j, ret;



    for (ch = 0; ch < channels; ch++) {

        IIRChannel *iir = &s->iir[ch];

        double *topc, *botc;



        topc = av_calloc((iir->nb_ab[0] + 1) * 2, sizeof(*topc));

        botc = av_calloc((iir->nb_ab[1] + 1) * 2, sizeof(*botc));

        if (!topc || !botc)

            return AVERROR(ENOMEM);



        ret = expand(ctx, iir->ab[0], iir->nb_ab[0], botc);

        if (ret < 0) {

            av_free(topc);

            av_free(botc);

            return ret;

        }



        ret = expand(ctx, iir->ab[1], iir->nb_ab[1], topc);

        if (ret < 0) {

            av_free(topc);

            av_free(botc);

            return ret;

        }



        for (j = 0, i = iir->nb_ab[1]; i >= 0; j++, i--) {

            iir->ab[1][j] = topc[2 * i];

        }

        iir->nb_ab[1]++;



        for (j = 0, i = iir->nb_ab[0]; i >= 0; j++, i--) {

            iir->ab[0][j] = botc[2 * i];

        }

        iir->nb_ab[0]++;



        av_free(topc);

        av_free(botc);

    }



    return 0;

}
