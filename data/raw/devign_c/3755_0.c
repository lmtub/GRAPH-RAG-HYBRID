void audio_pcm_init_info (struct audio_pcm_info *info, audsettings_t *as)

{

    int bits = 8, sign = 0, shift = 0;



    switch (as->fmt) {

    case AUD_FMT_S8:

        sign = 1;

    case AUD_FMT_U8:

        break;



    case AUD_FMT_S16:

        sign = 1;

    case AUD_FMT_U16:

        bits = 16;

        shift = 1;

        break;



    case AUD_FMT_S32:

        sign = 1;

    case AUD_FMT_U32:

        bits = 32;

        shift = 2;

        break;

    }



    info->freq = as->freq;

    info->bits = bits;

    info->sign = sign;

    info->nchannels = as->nchannels;

    info->shift = (as->nchannels == 2) + shift;

    info->align = (1 << info->shift) - 1;

    info->bytes_per_second = info->freq << info->shift;

    info->swap_endianness = (as->endianness != AUDIO_HOST_ENDIANNESS);

}
