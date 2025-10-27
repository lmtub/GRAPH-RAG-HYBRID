static void print_samplesref(AVFilterBufferRef *samplesref)

{

    const AVFilterBufferRefAudioProps *props = samplesref->audio;

    const int n = props->nb_samples * av_get_channel_layout_nb_channels(props->channel_layout);

    const uint16_t *p     = (uint16_t*)samplesref->data[0];

    const uint16_t *p_end = p + n;



    while (p < p_end) {

        fputc(*p    & 0xff, stdout);

        fputc(*p>>8 & 0xff, stdout);

        p++;

    }

    fflush(stdout);

}
