static int raw_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    int ret, size, bps;

    //    AVStream *st = s->streams[0];



    size= RAW_SAMPLES*s->streams[0]->codec->block_align;



    ret= av_get_packet(s->pb, pkt, size);




    pkt->stream_index = 0;

    if (ret < 0)

        return ret;



    bps= av_get_bits_per_sample(s->streams[0]->codec->codec_id);

    assert(bps); // if false there IS a bug elsewhere (NOT in this function)

    pkt->dts=

    pkt->pts= pkt->pos*8 / (bps * s->streams[0]->codec->channels);



    return ret;

}