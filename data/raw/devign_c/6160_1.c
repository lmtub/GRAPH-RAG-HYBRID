static int read_ffserver_streams(AVFormatContext *s, const char *filename)

{

    int i, err;

    AVFormatContext *ic;

    int nopts = 0;



    err = av_open_input_file(&ic, filename, NULL, FFM_PACKET_SIZE, NULL);

    if (err < 0)

        return err;

    /* copy stream format */

    s->nb_streams = ic->nb_streams;

    for(i=0;i<ic->nb_streams;i++) {

        AVStream *st;

        AVCodec *codec;



        // FIXME: a more elegant solution is needed

        st = av_mallocz(sizeof(AVStream));

        memcpy(st, ic->streams[i], sizeof(AVStream));

        st->codec = avcodec_alloc_context();

        if (!st->codec) {

            print_error(filename, AVERROR(ENOMEM));

            ffmpeg_exit(1);

        }

        avcodec_copy_context(st->codec, ic->streams[i]->codec);

        s->streams[i] = st;



        codec = avcodec_find_encoder(st->codec->codec_id);

        if (st->codec->codec_type == AVMEDIA_TYPE_AUDIO) {

            if (audio_stream_copy) {

                st->stream_copy = 1;

            } else

                choose_sample_fmt(st, codec);

        } else if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO) {

            if (video_stream_copy) {

                st->stream_copy = 1;

            } else

                choose_pixel_fmt(st, codec);

        }



        if(!st->codec->thread_count)

            st->codec->thread_count = 1;

        if(st->codec->thread_count>1)

            avcodec_thread_init(st->codec, st->codec->thread_count);



        if(st->codec->flags & CODEC_FLAG_BITEXACT)

            nopts = 1;

    }



    if (!nopts)

        s->timestamp = av_gettime();



    av_close_input_file(ic);

    return 0;

}
