static int wv_read_packet(AVFormatContext *s,

                          AVPacket *pkt)

{

    WVContext *wc = s->priv_data;

    int ret, samples;



    if (url_feof(&s->pb))

        return -EIO;

    if(wc->block_parsed){

        if(wv_read_block_header(s, &s->pb) < 0)

            return -1;

    }



    samples = LE_32(wc->extra);

    /* should not happen but who knows */

    if(samples * 2 * wc->chan > AVCODEC_MAX_AUDIO_FRAME_SIZE){

        av_log(s, AV_LOG_ERROR, "Packet size is too big to be handled in lavc!\n");

        return -EIO;

    }

    if(av_new_packet(pkt, wc->blksize + WV_EXTRA_SIZE) < 0)

        return AVERROR_NOMEM;

    memcpy(pkt->data, wc->extra, WV_EXTRA_SIZE);

    ret = get_buffer(&s->pb, pkt->data + WV_EXTRA_SIZE, wc->blksize);

    if(ret != wc->blksize){

        av_free_packet(pkt);

        return AVERROR_IO;

    }

    pkt->stream_index = 0;

    wc->block_parsed = 1;

    pkt->size = ret + WV_EXTRA_SIZE;



    return 0;

}
