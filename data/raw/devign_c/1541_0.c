int avformat_queue_attached_pictures(AVFormatContext *s)

{

    int i;

    for (i = 0; i < s->nb_streams; i++)

        if (s->streams[i]->disposition & AV_DISPOSITION_ATTACHED_PIC &&

            s->streams[i]->discard < AVDISCARD_ALL) {

            AVPacket copy = s->streams[i]->attached_pic;

            if (copy.size <= 0)

                return AVERROR(EINVAL);

            copy.buf = av_buffer_ref(copy.buf);

            if (!copy.buf)

                return AVERROR(ENOMEM);



            add_to_pktbuf(&s->raw_packet_buffer, &copy,

                          &s->raw_packet_buffer_end);

        }

    return 0;

}
