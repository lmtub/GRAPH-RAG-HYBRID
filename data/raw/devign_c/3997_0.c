static int amr_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    AVCodecContext *enc = s->streams[0]->codec;

    int read, size = 0, toc, mode;

    int64_t pos = avio_tell(s->pb);



    if (url_feof(s->pb)) {

        return AVERROR(EIO);

    }



    // FIXME this is wrong, this should rather be in a AVParset

    toc  = avio_r8(s->pb);

    mode = (toc >> 3) & 0x0F;



    if (enc->codec_id == AV_CODEC_ID_AMR_NB) {

        static const uint8_t packed_size[16] = {

            12, 13, 15, 17, 19, 20, 26, 31, 5, 0, 0, 0, 0, 0, 0, 0

        };



        size = packed_size[mode] + 1;

    } else if (enc->codec_id == AV_CODEC_ID_AMR_WB) {

        static const uint8_t packed_size[16] = {

            18, 24, 33, 37, 41, 47, 51, 59, 61, 6, 6, 0, 0, 0, 1, 1

        };



        size = packed_size[mode];

    } else {

        av_assert0(0);

    }



    if (!size || av_new_packet(pkt, size))

        return AVERROR(EIO);



    /* Both AMR formats have 50 frames per second */

    s->streams[0]->codec->bit_rate = size*8*50;



    pkt->stream_index = 0;

    pkt->pos          = pos;

    pkt->data[0]      = toc;

    pkt->duration     = enc->codec_id == AV_CODEC_ID_AMR_NB ? 160 : 320;

    read              = avio_read(s->pb, pkt->data + 1, size - 1);



    if (read != size - 1) {

        av_free_packet(pkt);

        return AVERROR(EIO);

    }



    return 0;

}
