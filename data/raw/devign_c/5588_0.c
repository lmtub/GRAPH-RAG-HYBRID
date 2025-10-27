static int latm_write_packet(AVFormatContext *s, AVPacket *pkt)

{

    LATMContext *ctx = s->priv_data;

    AVIOContext *pb = s->pb;

    PutBitContext bs;

    int i, len;

    uint8_t loas_header[] = "\x56\xe0\x00";



    if (s->streams[0]->codecpar->codec_id == AV_CODEC_ID_AAC_LATM)

        return ff_raw_write_packet(s, pkt);



    if (!s->streams[0]->codecpar->extradata) {

        if(pkt->size > 2 && pkt->data[0] == 0x56 && (pkt->data[1] >> 4) == 0xe &&

            (AV_RB16(pkt->data + 1) & 0x1FFF) + 3 == pkt->size)

            return ff_raw_write_packet(s, pkt);

        else

            return AVERROR_INVALIDDATA;

    }



    if (pkt->size > 0x1fff)

        goto too_large;



    init_put_bits(&bs, ctx->buffer, pkt->size+1024+MAX_EXTRADATA_SIZE);



    latm_write_frame_header(s, &bs);



    /* PayloadLengthInfo() */

    for (i = 0; i <= pkt->size-255; i+=255)

        put_bits(&bs, 8, 255);



    put_bits(&bs, 8, pkt->size-i);



    /* The LATM payload is written unaligned */



    /* PayloadMux() */

    if (pkt->size && (pkt->data[0] & 0xe1) == 0x81) {

        // Convert byte-aligned DSE to non-aligned.

        // Due to the input format encoding we know that

        // it is naturally byte-aligned in the input stream,

        // so there are no padding bits to account for.

        // To avoid having to add padding bits and rearrange

        // the whole stream we just remove the byte-align flag.

        // This allows us to remux our FATE AAC samples into latm

        // files that are still playable with minimal effort.

        put_bits(&bs, 8, pkt->data[0] & 0xfe);

        avpriv_copy_bits(&bs, pkt->data + 1, 8*pkt->size - 8);

    } else

        avpriv_copy_bits(&bs, pkt->data, 8*pkt->size);



    avpriv_align_put_bits(&bs);

    flush_put_bits(&bs);



    len = put_bits_count(&bs) >> 3;



    if (len > 0x1fff)

        goto too_large;



    loas_header[1] |= (len >> 8) & 0x1f;

    loas_header[2] |= len & 0xff;



    avio_write(pb, loas_header, 3);

    avio_write(pb, ctx->buffer, len);



    return 0;



too_large:

    av_log(s, AV_LOG_ERROR, "LATM packet size larger than maximum size 0x1fff\n");

    return AVERROR_INVALIDDATA;

}
