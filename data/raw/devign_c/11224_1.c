static int smush_read_packet(AVFormatContext *ctx, AVPacket *pkt)

{

    SMUSHContext *smush = ctx->priv_data;

    AVIOContext *pb = ctx->pb;

    int done = 0;



    while (!done) {

        uint32_t sig, size;



        if (url_feof(pb))

            return AVERROR_EOF;



        sig    = avio_rb32(pb);

        size   = avio_rb32(pb);



        switch (sig) {

        case MKBETAG('F', 'R', 'M', 'E'):

            if (smush->version)

                break;

            if (av_get_packet(pb, pkt, size) < 0)

                return AVERROR(EIO);



            pkt->stream_index = smush->video_stream_index;

            done = 1;

            break;

        case MKBETAG('B', 'l', '1', '6'):

            if (av_get_packet(pb, pkt, size) < 0)

                return AVERROR(EIO);



            pkt->stream_index = smush->video_stream_index;

            pkt->duration = 1;

            done = 1;

            break;

        case MKBETAG('W', 'a', 'v', 'e'):

            if (size < 13)

                return AVERROR_INVALIDDATA;

            if (av_get_packet(pb, pkt, size) < 0)

                return AVERROR(EIO);



            pkt->stream_index = smush->audio_stream_index;

            pkt->flags       |= AV_PKT_FLAG_KEY;

            pkt->duration = AV_RB32(pkt->data);

            if (pkt->duration == 0xFFFFFFFFu)

                pkt->duration = AV_RB32(pkt->data + 8);

            done = 1;

            break;

        default:

            avio_skip(pb, size);

            break;

        }

    }



    return 0;

}
