static int bfi_read_packet(AVFormatContext * s, AVPacket * pkt)

{

    BFIContext *bfi = s->priv_data;

    AVIOContext *pb = s->pb;

    int ret, audio_offset, video_offset, chunk_size, audio_size = 0;

    if (bfi->nframes == 0 || pb->eof_reached) {

        return AVERROR(EIO);




    /* If all previous chunks were completely read, then find a new one... */

    if (!bfi->avflag) {

        uint32_t state = 0;

        while(state != MKTAG('S','A','V','I')){

            if (pb->eof_reached)

                return AVERROR(EIO);

            state = 256*state + avio_r8(pb);


        /* Now that the chunk's location is confirmed, we proceed... */

        chunk_size      = avio_rl32(pb);

        avio_rl32(pb);

        audio_offset    = avio_rl32(pb);

        avio_rl32(pb);

        video_offset    = avio_rl32(pb);

        audio_size      = video_offset - audio_offset;

        bfi->video_size = chunk_size - video_offset;







        //Tossing an audio packet at the audio decoder.

        ret = av_get_packet(pb, pkt, audio_size);

        if (ret < 0)

            return ret;



        pkt->pts          = bfi->audio_frame;

        bfi->audio_frame += ret;

    } else if (bfi->video_size > 0) {



        //Tossing a video packet at the video decoder.

        ret = av_get_packet(pb, pkt, bfi->video_size);

        if (ret < 0)

            return ret;



        pkt->pts          = bfi->video_frame;

        bfi->video_frame += ret / bfi->video_size;



        /* One less frame to read. A cursory decrement. */

        bfi->nframes--;

    } else {

        /* Empty video packet */

        ret = AVERROR(EAGAIN);




    bfi->avflag       = !bfi->avflag;

    pkt->stream_index = bfi->avflag;

    return ret;
