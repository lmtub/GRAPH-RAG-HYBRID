static av_cold int channelmap_init(AVFilterContext *ctx)

{

    ChannelMapContext *s = ctx->priv;

    int ret;

    char *mapping, separator = '|';

    int map_entries = 0;

    char buf[256];

    enum MappingMode mode;

    uint64_t out_ch_mask = 0;

    int i;



    mapping = s->mapping_str;



    if (!mapping) {

        mode = MAP_NONE;

    } else {

        char *dash = strchr(mapping, '-');

        if (!dash) {  // short mapping

            if (av_isdigit(*mapping))

                mode = MAP_ONE_INT;

            else

                mode = MAP_ONE_STR;

        } else if (av_isdigit(*mapping)) {

            if (av_isdigit(*(dash+1)))

                mode = MAP_PAIR_INT_INT;

            else

                mode = MAP_PAIR_INT_STR;

        } else {

            if (av_isdigit(*(dash+1)))

                mode = MAP_PAIR_STR_INT;

            else

                mode = MAP_PAIR_STR_STR;

        }

#if FF_API_OLD_FILTER_OPTS

        if (strchr(mapping, ',')) {

            av_log(ctx, AV_LOG_WARNING, "This syntax is deprecated, use "

                   "'|' to separate the mappings.\n");

            separator = ',';

        }

#endif

    }



    if (mode != MAP_NONE) {

        char *sep = mapping;

        map_entries = 1;

        while ((sep = strchr(sep, separator))) {

            if (*++sep)  // Allow trailing comma

                map_entries++;

        }

    }



    if (map_entries > MAX_CH) {

        av_log(ctx, AV_LOG_ERROR, "Too many channels mapped: '%d'.\n", map_entries);

        ret = AVERROR(EINVAL);

        goto fail;

    }



    for (i = 0; i < map_entries; i++) {

        int in_ch_idx = -1, out_ch_idx = -1;

        uint64_t in_ch = 0, out_ch = 0;

        static const char err[] = "Failed to parse channel map\n";

        switch (mode) {

        case MAP_ONE_INT:

            if (get_channel_idx(&mapping, &in_ch_idx, separator, MAX_CH) < 0) {

                ret = AVERROR(EINVAL);

                av_log(ctx, AV_LOG_ERROR, err);

                goto fail;

            }

            s->map[i].in_channel_idx  = in_ch_idx;

            s->map[i].out_channel_idx = i;

            break;

        case MAP_ONE_STR:

            if (!get_channel(&mapping, &in_ch, separator)) {

                av_log(ctx, AV_LOG_ERROR, err);

                ret = AVERROR(EINVAL);

                goto fail;

            }

            s->map[i].in_channel      = in_ch;

            s->map[i].out_channel_idx = i;

            break;

        case MAP_PAIR_INT_INT:

            if (get_channel_idx(&mapping, &in_ch_idx, '-', MAX_CH) < 0 ||

                get_channel_idx(&mapping, &out_ch_idx, separator, MAX_CH) < 0) {

                av_log(ctx, AV_LOG_ERROR, err);

                ret = AVERROR(EINVAL);

                goto fail;

            }

            s->map[i].in_channel_idx  = in_ch_idx;

            s->map[i].out_channel_idx = out_ch_idx;

            break;

        case MAP_PAIR_INT_STR:

            if (get_channel_idx(&mapping, &in_ch_idx, '-', MAX_CH) < 0 ||

                get_channel(&mapping, &out_ch, separator) < 0 ||

                out_ch & out_ch_mask) {

                av_log(ctx, AV_LOG_ERROR, err);

                ret = AVERROR(EINVAL);

                goto fail;

            }

            s->map[i].in_channel_idx  = in_ch_idx;

            s->map[i].out_channel     = out_ch;

            out_ch_mask |= out_ch;

            break;

        case MAP_PAIR_STR_INT:

            if (get_channel(&mapping, &in_ch, '-') < 0 ||

                get_channel_idx(&mapping, &out_ch_idx, separator, MAX_CH) < 0) {

                av_log(ctx, AV_LOG_ERROR, err);

                ret = AVERROR(EINVAL);

                goto fail;

            }

            s->map[i].in_channel      = in_ch;

            s->map[i].out_channel_idx = out_ch_idx;

            break;

        case MAP_PAIR_STR_STR:

            if (get_channel(&mapping, &in_ch, '-') < 0 ||

                get_channel(&mapping, &out_ch, separator) < 0 ||

                out_ch & out_ch_mask) {

                av_log(ctx, AV_LOG_ERROR, err);

                ret = AVERROR(EINVAL);

                goto fail;

            }

            s->map[i].in_channel = in_ch;

            s->map[i].out_channel = out_ch;

            out_ch_mask |= out_ch;

            break;

        }

    }

    s->mode          = mode;

    s->nch           = map_entries;

    s->output_layout = out_ch_mask ? out_ch_mask :

                       av_get_default_channel_layout(map_entries);



    if (s->channel_layout_str) {

        uint64_t fmt;

        if ((fmt = av_get_channel_layout(s->channel_layout_str)) == 0) {

            av_log(ctx, AV_LOG_ERROR, "Error parsing channel layout: '%s'.\n",

                   s->channel_layout_str);

            ret = AVERROR(EINVAL);

            goto fail;

        }

        if (mode == MAP_NONE) {

            int i;

            s->nch = av_get_channel_layout_nb_channels(fmt);

            for (i = 0; i < s->nch; i++) {

                s->map[i].in_channel_idx  = i;

                s->map[i].out_channel_idx = i;

            }

        } else if (out_ch_mask && out_ch_mask != fmt) {

            av_get_channel_layout_string(buf, sizeof(buf), 0, out_ch_mask);

            av_log(ctx, AV_LOG_ERROR,

                   "Output channel layout '%s' does not match the list of channel mapped: '%s'.\n",

                   s->channel_layout_str, buf);

            ret = AVERROR(EINVAL);

            goto fail;

        } else if (s->nch != av_get_channel_layout_nb_channels(fmt)) {

            av_log(ctx, AV_LOG_ERROR,

                   "Output channel layout %s does not match the number of channels mapped %d.\n",

                   s->channel_layout_str, s->nch);

            ret = AVERROR(EINVAL);

            goto fail;

        }

        s->output_layout = fmt;

    }

    ff_add_channel_layout(&s->channel_layouts, s->output_layout);



    if (mode == MAP_PAIR_INT_STR || mode == MAP_PAIR_STR_STR) {

        for (i = 0; i < s->nch; i++) {

            s->map[i].out_channel_idx = av_get_channel_layout_channel_index(

                s->output_layout, s->map[i].out_channel);

        }

    }



fail:

    av_opt_free(s);

    return ret;

}
