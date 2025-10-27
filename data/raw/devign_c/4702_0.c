static int mov_write_identification(AVIOContext *pb, AVFormatContext *s)

{

    MOVMuxContext *mov = s->priv_data;

    int i;



    mov_write_ftyp_tag(pb,s);

    if (mov->mode == MODE_PSP) {

        int video_streams_nb = 0, audio_streams_nb = 0, other_streams_nb = 0;

        for (i = 0; i < s->nb_streams; i++) {

            AVStream *st = s->streams[i];

            if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)

                video_streams_nb++;

            else if (st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO)

                audio_streams_nb++;

            else

                other_streams_nb++;

            }



        if (video_streams_nb != 1 || audio_streams_nb != 1 || other_streams_nb) {

            av_log(s, AV_LOG_ERROR, "PSP mode need one video and one audio stream\n");

            return AVERROR(EINVAL);

        }

        mov_write_uuidprof_tag(pb, s);

    }

    return 0;

}
