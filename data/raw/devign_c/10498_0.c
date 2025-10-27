static int dxva2_retrieve_data(AVCodecContext *s, AVFrame *frame)

{

    InputStream        *ist = s->opaque;

    DXVA2Context       *ctx = ist->hwaccel_ctx;

    int                ret;



    ret = av_hwframe_transfer_data(ctx->tmp_frame, frame, 0);

    if (ret < 0)

        return ret;



    ret = av_frame_copy_props(ctx->tmp_frame, frame);

    if (ret < 0) {

        av_frame_unref(ctx->tmp_frame);

        return ret;

    }



    av_frame_unref(frame);

    av_frame_move_ref(frame, ctx->tmp_frame);



    return 0;

}
