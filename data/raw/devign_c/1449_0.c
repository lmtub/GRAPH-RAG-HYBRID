static int film_read_close(AVFormatContext *s)

{

    FilmDemuxContext *film = s->priv_data;



    av_freep(&film->sample_table);

    av_freep(&film->stereo_buffer);



    return 0;

}
