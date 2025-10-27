static int mpeg_mux_end(AVFormatContext *ctx)

{

    MpegMuxContext *s = ctx->priv_data;

    StreamInfo *stream;

    int i;



    /* flush each packet */

    for(i=0;i<ctx->nb_streams;i++) {

        stream = ctx->streams[i]->priv_data;

        while (stream->buffer_ptr > 0) {

            flush_packet(ctx, i, AV_NOPTS_VALUE, AV_NOPTS_VALUE, s->last_scr);

        }

    }



    /* End header according to MPEG1 systems standard. We do not write

       it as it is usually not needed by decoders and because it

       complicates MPEG stream concatenation. */

    //put_be32(&ctx->pb, ISO_11172_END_CODE);

    //put_flush_packet(&ctx->pb);



    for(i=0;i<ctx->nb_streams;i++)

        av_freep(&ctx->streams[i]->priv_data);



    return 0;

}
