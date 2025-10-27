int read_ffserver_streams(AVFormatContext *s, const char *filename)

{

    int i;

    AVFormatContext *ic;



    ic = av_open_input_file(filename, FFM_PACKET_SIZE);

    if (!ic)

        return -EIO;

    /* copy stream format */

    s->nb_streams = ic->nb_streams;

    for(i=0;i<ic->nb_streams;i++) {

        AVStream *st;

        st = av_mallocz(sizeof(AVFormatContext));

        memcpy(st, ic->streams[i], sizeof(AVStream));

        s->streams[i] = st;

    }



    av_close_input_file(ic);

    return 0;

}
