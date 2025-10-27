static int amr_read_packet(AVFormatContext *s,

                          AVPacket *pkt)

{

    AVCodecContext *enc = s->streams[0]->codec;

    int read, size, toc, mode;



    if (url_feof(&s->pb))

    {

        return AVERROR_IO;

    }



//FIXME this is wrong, this should rather be in a AVParset

    toc=get_byte(&s->pb);

    mode = (toc >> 3) & 0x0F;



    if (enc->codec_id == CODEC_ID_AMR_NB)

    {

        static const uint8_t packed_size[16] = {12, 13, 15, 17, 19, 20, 26, 31, 5, 0, 0, 0, 0, 0, 0, 0};



        size=packed_size[mode]+1;

    }

    else if(enc->codec_id == CODEC_ID_AMR_WB)

    {

        static uint8_t packed_size[16] = {18, 24, 33, 37, 41, 47, 51, 59, 61, 6, 6, 0, 0, 0, 1, 1};



        size=packed_size[mode];

    }

    else

    {

        assert(0);

    }



    if ( (size==0) || av_new_packet(pkt, size))

    {

        return AVERROR_IO;

    }



    pkt->stream_index = 0;

    pkt->pos= url_ftell(&s->pb);

    pkt->data[0]=toc;

    pkt->duration= enc->codec_id == CODEC_ID_AMR_NB ? 160 : 320;

    read = get_buffer(&s->pb, pkt->data+1, size-1);



    if (read != size-1)

    {

        av_free_packet(pkt);

        return AVERROR_IO;

    }



    return 0;

}
