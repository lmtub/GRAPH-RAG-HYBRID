static void qsv_uninit(AVCodecContext *s)

{

    InputStream *ist = s->opaque;

    QSVContext  *qsv = ist->hwaccel_ctx;



    av_freep(&qsv->ost->enc_ctx->hwaccel_context);

    av_freep(&s->hwaccel_context);



    av_buffer_unref(&qsv->opaque_surfaces_buf);

    av_freep(&qsv->surface_used);

    av_freep(&qsv->surface_ptrs);



    av_freep(&qsv);

}
