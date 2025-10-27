static av_cold int set_channel_info(AC3EncodeContext *s, int channels,

                                    int64_t *channel_layout)

{

    int ch_layout;



    if (channels < 1 || channels > AC3_MAX_CHANNELS)

        return AVERROR(EINVAL);

    if ((uint64_t)*channel_layout > 0x7FF)

        return AVERROR(EINVAL);

    ch_layout = *channel_layout;

    if (!ch_layout)

        ch_layout = avcodec_guess_channel_layout(channels, CODEC_ID_AC3, NULL);

    if (av_get_channel_layout_nb_channels(ch_layout) != channels)

        return AVERROR(EINVAL);



    s->lfe_on       = !!(ch_layout & AV_CH_LOW_FREQUENCY);

    s->channels     = channels;

    s->fbw_channels = channels - s->lfe_on;

    s->lfe_channel  = s->lfe_on ? s->fbw_channels : -1;

    if (s->lfe_on)

        ch_layout -= AV_CH_LOW_FREQUENCY;



    switch (ch_layout) {

    case AV_CH_LAYOUT_MONO:           s->channel_mode = AC3_CHMODE_MONO;   break;

    case AV_CH_LAYOUT_STEREO:         s->channel_mode = AC3_CHMODE_STEREO; break;

    case AV_CH_LAYOUT_SURROUND:       s->channel_mode = AC3_CHMODE_3F;     break;

    case AV_CH_LAYOUT_2_1:            s->channel_mode = AC3_CHMODE_2F1R;   break;

    case AV_CH_LAYOUT_4POINT0:        s->channel_mode = AC3_CHMODE_3F1R;   break;

    case AV_CH_LAYOUT_QUAD:

    case AV_CH_LAYOUT_2_2:            s->channel_mode = AC3_CHMODE_2F2R;   break;

    case AV_CH_LAYOUT_5POINT0:

    case AV_CH_LAYOUT_5POINT0_BACK:   s->channel_mode = AC3_CHMODE_3F2R;   break;

    default:

        return AVERROR(EINVAL);

    }

    s->has_center   = (s->channel_mode & 0x01) && s->channel_mode != AC3_CHMODE_MONO;

    s->has_surround =  s->channel_mode & 0x04;



    s->channel_map  = ff_ac3_enc_channel_map[s->channel_mode][s->lfe_on];

    *channel_layout = ch_layout;

    if (s->lfe_on)

        *channel_layout |= AV_CH_LOW_FREQUENCY;



    return 0;

}
