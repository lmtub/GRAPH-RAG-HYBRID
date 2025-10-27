av_cold int ff_opus_parse_extradata(AVCodecContext *avctx,

                                    OpusContext *s)

{

    static const uint8_t default_channel_map[2] = { 0, 1 };



    int (*channel_reorder)(int, int) = channel_reorder_unknown;



    const uint8_t *extradata, *channel_map;

    int extradata_size;

    int version, channels, map_type, streams, stereo_streams, i, j;

    uint64_t layout;



    if (!avctx->extradata) {

        if (avctx->channels > 2) {

            av_log(avctx, AV_LOG_ERROR,

                   "Multichannel configuration without extradata.\n");

            return AVERROR(EINVAL);

        }

        extradata      = opus_default_extradata;

        extradata_size = sizeof(opus_default_extradata);

    } else {

        extradata = avctx->extradata;

        extradata_size = avctx->extradata_size;

    }



    if (extradata_size < 19) {

        av_log(avctx, AV_LOG_ERROR, "Invalid extradata size: %d\n",

               extradata_size);

        return AVERROR_INVALIDDATA;

    }



    version = extradata[8];

    if (version > 15) {

        avpriv_request_sample(avctx, "Extradata version %d", version);

        return AVERROR_PATCHWELCOME;

    }



    avctx->delay = AV_RL16(extradata + 10);



    channels = avctx->extradata ? extradata[9] : (avctx->channels == 1) ? 1 : 2;

    if (!channels) {

        av_log(avctx, AV_LOG_ERROR, "Zero channel count specified in the extradata\n");

        return AVERROR_INVALIDDATA;

    }



    s->gain_i = AV_RL16(extradata + 16);

    if (s->gain_i)

        s->gain = ff_exp10(s->gain_i / (20.0 * 256));



    map_type = extradata[18];

    if (!map_type) {

        if (channels > 2) {

            av_log(avctx, AV_LOG_ERROR,

                   "Channel mapping 0 is only specified for up to 2 channels\n");

            return AVERROR_INVALIDDATA;

        }

        layout         = (channels == 1) ? AV_CH_LAYOUT_MONO : AV_CH_LAYOUT_STEREO;

        streams        = 1;

        stereo_streams = channels - 1;

        channel_map    = default_channel_map;

    } else if (map_type == 1 || map_type == 2 || map_type == 255) {

        if (extradata_size < 21 + channels) {

            av_log(avctx, AV_LOG_ERROR, "Invalid extradata size: %d\n",

                   extradata_size);

            return AVERROR_INVALIDDATA;

        }



        streams        = extradata[19];

        stereo_streams = extradata[20];

        if (!streams || stereo_streams > streams ||

            streams + stereo_streams > 255) {

            av_log(avctx, AV_LOG_ERROR,

                   "Invalid stream/stereo stream count: %d/%d\n", streams, stereo_streams);

            return AVERROR_INVALIDDATA;

        }



        if (map_type == 1) {

            if (channels > 8) {

                av_log(avctx, AV_LOG_ERROR,

                       "Channel mapping 1 is only specified for up to 8 channels\n");

                return AVERROR_INVALIDDATA;

            }

            layout = ff_vorbis_channel_layouts[channels - 1];

            channel_reorder = channel_reorder_vorbis;

        } else if (map_type == 2) {

            int ambisonic_order = ff_sqrt(channels) - 1;

            if (channels != (ambisonic_order + 1) * (ambisonic_order + 1)) {

                av_log(avctx, AV_LOG_ERROR,

                       "Channel mapping 2 is only specified for channel counts"

                       " which can be written as (n + 1)^2 for nonnegative integer n\n");

                return AVERROR_INVALIDDATA;

            }

            layout = 0;

        } else

            layout = 0;



        channel_map = extradata + 21;

    } else {

        avpriv_request_sample(avctx, "Mapping type %d", map_type);

        return AVERROR_PATCHWELCOME;

    }



    s->channel_maps = av_mallocz_array(channels, sizeof(*s->channel_maps));

    if (!s->channel_maps)

        return AVERROR(ENOMEM);



    for (i = 0; i < channels; i++) {

        ChannelMap *map = &s->channel_maps[i];

        uint8_t     idx = channel_map[channel_reorder(channels, i)];



        if (idx == 255) {

            map->silence = 1;

            continue;

        } else if (idx >= streams + stereo_streams) {

            av_log(avctx, AV_LOG_ERROR,

                   "Invalid channel map for output channel %d: %d\n", i, idx);


            return AVERROR_INVALIDDATA;

        }



        /* check that we did not see this index yet */

        map->copy = 0;

        for (j = 0; j < i; j++)

            if (channel_map[channel_reorder(channels, j)] == idx) {

                map->copy     = 1;

                map->copy_idx = j;

                break;

            }



        if (idx < 2 * stereo_streams) {

            map->stream_idx  = idx / 2;

            map->channel_idx = idx & 1;

        } else {

            map->stream_idx  = idx - stereo_streams;

            map->channel_idx = 0;

        }

    }



    avctx->channels       = channels;

    avctx->channel_layout = layout;

    s->nb_streams         = streams;

    s->nb_stereo_streams  = stereo_streams;



    return 0;

}