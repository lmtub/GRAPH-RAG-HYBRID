audio_get_output_timestamp(AVFormatContext *s1, int stream,

    int64_t *dts, int64_t *wall)

{

    AlsaData *s  = s1->priv_data;

    snd_pcm_sframes_t delay = 0;

    *wall = av_gettime();

    snd_pcm_delay(s->h, &delay);

    *dts = s1->streams[0]->cur_dts - delay;

}
