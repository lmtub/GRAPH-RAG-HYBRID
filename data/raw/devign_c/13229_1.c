static double get_scene_score(AVFilterContext *ctx, AVFilterBufferRef *picref)

{

    double ret = 0;

    SelectContext *select = ctx->priv;

    AVFilterBufferRef *prev_picref = select->prev_picref;



    if (prev_picref &&

        picref->video->h    == prev_picref->video->h &&

        picref->video->w    == prev_picref->video->w &&

        picref->linesize[0] == prev_picref->linesize[0]) {

        int x, y;

        int64_t sad = 0;

        double mafd, diff;

        uint8_t *p1 =      picref->data[0];

        uint8_t *p2 = prev_picref->data[0];

        const int linesize = picref->linesize[0];



        for (y = 0; y < picref->video->h; y += 8)

            for (x = 0; x < linesize; x += 8)

                sad += select->c.sad[1](select,

                                        p1 + y * linesize + x,

                                        p2 + y * linesize + x,

                                        linesize, 8);

        emms_c();

        mafd = sad / (picref->video->h * picref->video->w * 3);

        diff = fabs(mafd - select->prev_mafd);

        ret  = av_clipf(FFMIN(mafd, diff) / 100., 0, 1);

        select->prev_mafd = mafd;

        avfilter_unref_buffer(prev_picref);

    }

    select->prev_picref = avfilter_ref_buffer(picref, ~0);

    return ret;

}
