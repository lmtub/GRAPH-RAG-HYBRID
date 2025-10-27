static int audio_pcm_info_eq (struct audio_pcm_info *info, audsettings_t *as)

{

    int bits = 8, sign = 0;



    switch (as->fmt) {

    case AUD_FMT_S8:

        sign = 1;

    case AUD_FMT_U8:

        break;



    case AUD_FMT_S16:

        sign = 1;

    case AUD_FMT_U16:

        bits = 16;

        break;



    case AUD_FMT_S32:

        sign = 1;

    case AUD_FMT_U32:

        bits = 32;

        break;

    }

    return info->freq == as->freq

        && info->nchannels == as->nchannels

        && info->sign == sign

        && info->bits == bits

        && info->swap_endianness == (as->endianness != AUDIO_HOST_ENDIANNESS);

}
