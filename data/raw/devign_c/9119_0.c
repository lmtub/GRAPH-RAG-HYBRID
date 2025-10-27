static void draw_curves(AVFilterContext *ctx, AVFilterLink *inlink, AVFrame *out)

{

    AudioNEqualizerContext *s = ctx->priv;

    char *colors, *color, *saveptr = NULL;

    int ch, i, n;



    colors = av_strdup(s->colors);

    if (!colors)

        return;



    memset(out->data[0], 0, s->h * out->linesize[0]);



    for (ch = 0; ch < inlink->channels; ch++) {

        uint8_t fg[4] = { 0xff, 0xff, 0xff, 0xff };

        int prev_v = -1;

        double f;



        color = av_strtok(ch == 0 ? colors : NULL, " |", &saveptr);

        if (color)

            av_parse_color(fg, color, -1, ctx);



        for (f = 0; f < s->w; f++) {

            double complex z;

            double complex H = 1;

            double w;

            int v, y, x;



            w = M_PI * (s->fscale ? pow(s->w - 1, f / s->w) : f) / (s->w - 1);

            z = 1. / cexp(I * w);



            for (n = 0; n < s->nb_filters; n++) {

                if (s->filters[n].channel != ch ||

                    s->filters[n].ignore)

                    continue;



                for (i = 0; i < FILTER_ORDER / 2; i++) {

                    FoSection *S = &s->filters[n].section[i];



                    H *= (((((S->b4 * z + S->b3) * z + S->b2) * z + S->b1) * z + S->b0) /

                          ((((S->a4 * z + S->a3) * z + S->a2) * z + S->a1) * z + S->a0));

                }

            }



            v = av_clip((1. + -20 * log10(cabs(H)) / s->mag) * s->h / 2, 0, s->h - 1);

            x = lrint(f);

            if (prev_v == -1)

                prev_v = v;

            if (v <= prev_v) {

                for (y = v; y <= prev_v; y++)

                    AV_WL32(out->data[0] + y * out->linesize[0] + x * 4, AV_RL32(fg));

            } else {

                for (y = prev_v; y <= v; y++)

                    AV_WL32(out->data[0] + y * out->linesize[0] + x * 4, AV_RL32(fg));

            }



            prev_v = v;

        }

    }



    av_free(colors);

}
