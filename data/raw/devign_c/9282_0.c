static int request_frame(AVFilterLink *link)

{

    AVFilterContext *ctx = link->src;

    IDETContext *idet = ctx->priv;



    do {

        int ret;



        if (idet->eof)

            return AVERROR_EOF;



        ret = ff_request_frame(link->src->inputs[0]);



        if (ret == AVERROR_EOF && idet->cur) {

            AVFrame *next = av_frame_clone(idet->next);



            if (!next)

                return AVERROR(ENOMEM);



            filter_frame(link->src->inputs[0], next);

            idet->eof = 1;

        } else if (ret < 0) {

            return ret;

        }

    } while (!idet->cur);



    return 0;

}
