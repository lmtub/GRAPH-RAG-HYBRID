static int spdif_write_packet(struct AVFormatContext *s, AVPacket *pkt)

{

    IEC958Context *ctx = s->priv_data;

    int ret, padding;




    ctx->out_bytes = pkt->size;

    ctx->length_code = FFALIGN(pkt->size, 2) << 3;



    ret = ctx->header_info(s, pkt);

    if (ret < 0)

        return -1;

    if (!ctx->pkt_offset)

        return 0;



    padding = (ctx->pkt_offset - BURST_HEADER_SIZE - ctx->out_bytes) >> 1;

    if (padding < 0) {

        av_log(s, AV_LOG_ERROR, "bitrate is too high\n");

        return -1;

    }



    put_le16(s->pb, SYNCWORD1);      //Pa

    put_le16(s->pb, SYNCWORD2);      //Pb

    put_le16(s->pb, ctx->data_type); //Pc

    put_le16(s->pb, ctx->length_code);//Pd



#if HAVE_BIGENDIAN

    put_buffer(s->pb, ctx->out_buf, ctx->out_bytes & ~1);

#else

    av_fast_malloc(&ctx->buffer, &ctx->buffer_size, ctx->out_bytes + FF_INPUT_BUFFER_PADDING_SIZE);

    if (!ctx->buffer)

        return AVERROR(ENOMEM);

    ff_spdif_bswap_buf16((uint16_t *)ctx->buffer, (uint16_t *)ctx->out_buf, ctx->out_bytes >> 1);

    put_buffer(s->pb, ctx->buffer, ctx->out_bytes & ~1);

#endif



    if (ctx->out_bytes & 1)

        put_be16(s->pb, ctx->out_buf[ctx->out_bytes - 1]);



    for (; padding > 0; padding--)

        put_be16(s->pb, 0);



    av_log(s, AV_LOG_DEBUG, "type=%x len=%i pkt_offset=%i\n",

           ctx->data_type, ctx->out_bytes, ctx->pkt_offset);



    put_flush_packet(s->pb);

    return 0;

}