static int mp3_write_packet(AVFormatContext *s, AVPacket *pkt)

{

    MP3Context *mp3 = s->priv_data;



    if (pkt->stream_index == mp3->audio_stream_idx) {

        if (mp3->pics_to_write) {

            /* buffer audio packets until we get all the pictures */

            AVPacketList *pktl = av_mallocz(sizeof(*pktl));

            if (!pktl)

                return AVERROR(ENOMEM);



            pktl->pkt     = *pkt;

            pktl->pkt.buf = av_buffer_ref(pkt->buf);

            if (!pktl->pkt.buf) {

                av_freep(&pktl);

                return AVERROR(ENOMEM);

            }



            if (mp3->queue_end)

                mp3->queue_end->next = pktl;

            else

                mp3->queue = pktl;

            mp3->queue_end = pktl;

        } else

            return mp3_write_audio_packet(s, pkt);

    } else {

        int ret;



        /* warn only once for each stream */

        if (s->streams[pkt->stream_index]->nb_frames == 1) {

            av_log(s, AV_LOG_WARNING, "Got more than one picture in stream %d,"

                   " ignoring.\n", pkt->stream_index);

        }

        if (!mp3->pics_to_write || s->streams[pkt->stream_index]->nb_frames >= 1)

            return 0;



        if ((ret = ff_id3v2_write_apic(s, &mp3->id3, pkt)) < 0)

            return ret;

        mp3->pics_to_write--;



        /* flush the buffered audio packets */

        if (!mp3->pics_to_write &&

            (ret = mp3_queue_flush(s)) < 0)

            return ret;

    }



    return 0;

}
