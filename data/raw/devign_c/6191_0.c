static int request_frame(AVFilterLink *outlink)

{

    AVFilterContext *ctx = outlink->src;

    TrimContext       *s = ctx->priv;

    int ret;



    s->got_output = 0;

    while (!s->got_output) {

        if (s->eof)

            return AVERROR_EOF;



        ret = ff_request_frame(ctx->inputs[0]);

        if (ret < 0)

            return ret;

    }



    return 0;

}
