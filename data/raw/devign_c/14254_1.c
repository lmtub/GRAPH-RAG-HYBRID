static int ape_read_packet(AVFormatContext * s, AVPacket * pkt)

{

    int ret;

    int nblocks;

    APEContext *ape = s->priv_data;

    uint32_t extra_size = 8;



    if (url_feof(s->pb))

        return AVERROR_EOF;

    if (ape->currentframe >= ape->totalframes)

        return AVERROR_EOF;



    if (avio_seek(s->pb, ape->frames[ape->currentframe].pos, SEEK_SET) < 0)

        return AVERROR(EIO);



    /* Calculate how many blocks there are in this frame */

    if (ape->currentframe == (ape->totalframes - 1))

        nblocks = ape->finalframeblocks;

    else

        nblocks = ape->blocksperframe;



    if (ape->frames[ape->currentframe].size <= 0 ||

        ape->frames[ape->currentframe].size > INT_MAX - extra_size) {

        av_log(s, AV_LOG_ERROR, "invalid packet size: %d\n",

               ape->frames[ape->currentframe].size);

        ape->currentframe++;

        return AVERROR(EIO);

    }



    if (av_new_packet(pkt,  ape->frames[ape->currentframe].size + extra_size) < 0)

        return AVERROR(ENOMEM);



    AV_WL32(pkt->data    , nblocks);

    AV_WL32(pkt->data + 4, ape->frames[ape->currentframe].skip);

    ret = avio_read(s->pb, pkt->data + extra_size, ape->frames[ape->currentframe].size);





    pkt->pts = ape->frames[ape->currentframe].pts;

    pkt->stream_index = 0;



    /* note: we need to modify the packet size here to handle the last

       packet */

    pkt->size = ret + extra_size;



    ape->currentframe++;



    return 0;

}