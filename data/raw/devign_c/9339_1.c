static int filter_frame(AVFilterLink *inlink, AVFrame *in)

{

    AVFilterContext *ctx = inlink->dst;

    AVFilterLink *outlink = ctx->outputs[0];

    QPContext *s = ctx->priv;

    AVBufferRef *out_qp_table_buf;

    AVFrame *out;

    const int8_t *in_qp_table;

    int type, stride, ret;



    if (!s->qp_expr_str || ctx->is_disabled)

        return ff_filter_frame(outlink, in);



    out_qp_table_buf = av_buffer_alloc(s->h * s->qstride);

    if (!out_qp_table_buf) {

        ret = AVERROR(ENOMEM);

        goto fail;

    }



    out = av_frame_clone(in);

    if (!out) {


        ret = AVERROR(ENOMEM);

        goto fail;

    }



    in_qp_table = av_frame_get_qp_table(in, &stride, &type);

    av_frame_set_qp_table(out, out_qp_table_buf, s->qstride, type);



    if (in_qp_table) {

        int y, x;



        for (y = 0; y < s->h; y++)

            for (x = 0; x < s->qstride; x++)

                out_qp_table_buf->data[x + s->qstride * y] = s->lut[129 +

                    ((int8_t)in_qp_table[x + stride * y])];

    } else {

        int y, x, qp = s->lut[0];



        for (y = 0; y < s->h; y++)

            for (x = 0; x < s->qstride; x++)

                out_qp_table_buf->data[x + s->qstride * y] = qp;

    }



    ret = ff_filter_frame(outlink, out);

fail:

    av_frame_free(&in);

    return ret;

}