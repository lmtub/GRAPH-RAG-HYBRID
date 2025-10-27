static av_cold int init(AVFilterContext *ctx, const char *args0)

{

    PanContext *const pan = ctx->priv;

    char *arg, *arg0, *tokenizer, *args = av_strdup(args0);

    int out_ch_id, in_ch_id, len, named, ret;

    int nb_in_channels[2] = { 0, 0 }; // number of unnamed and named input channels

    double gain;



    if (!args0) {

        av_log(ctx, AV_LOG_ERROR,

               "pan filter needs a channel layout and a set "

               "of channels definitions as parameter\n");

        return AVERROR(EINVAL);

    }

    if (!args)

        return AVERROR(ENOMEM);

    arg = av_strtok(args, ":", &tokenizer);

    ret = ff_parse_channel_layout(&pan->out_channel_layout, arg, ctx);

    if (ret < 0)

        return ret;

    pan->nb_output_channels = av_get_channel_layout_nb_channels(pan->out_channel_layout);



    /* parse channel specifications */

    while ((arg = arg0 = av_strtok(NULL, ":", &tokenizer))) {

        /* channel name */

        if (parse_channel_name(&arg, &out_ch_id, &named)) {

            av_log(ctx, AV_LOG_ERROR,

                   "Expected out channel name, got \"%.8s\"\n", arg);

            return AVERROR(EINVAL);

        }

        if (named) {

            if (!((pan->out_channel_layout >> out_ch_id) & 1)) {

                av_log(ctx, AV_LOG_ERROR,

                       "Channel \"%.8s\" does not exist in the chosen layout\n", arg0);

                return AVERROR(EINVAL);

            }

            /* get the channel number in the output channel layout:

             * out_channel_layout & ((1 << out_ch_id) - 1) are all the

             * channels that come before out_ch_id,

             * so their count is the index of out_ch_id */

            out_ch_id = av_get_channel_layout_nb_channels(pan->out_channel_layout & (((int64_t)1 << out_ch_id) - 1));

        }

        if (out_ch_id < 0 || out_ch_id >= pan->nb_output_channels) {

            av_log(ctx, AV_LOG_ERROR,

                   "Invalid out channel name \"%.8s\"\n", arg0);

            return AVERROR(EINVAL);

        }

        if (*arg == '=') {

            arg++;

        } else if (*arg == '<') {

            pan->need_renorm |= (int64_t)1 << out_ch_id;

            arg++;

        } else {

            av_log(ctx, AV_LOG_ERROR,

                   "Syntax error after channel name in \"%.8s\"\n", arg0);

            return AVERROR(EINVAL);

        }

        /* gains */

        while (1) {

            gain = 1;

            if (sscanf(arg, " %lf %n* %n", &gain, &len, &len))

                arg += len;

            if (parse_channel_name(&arg, &in_ch_id, &named)){

                av_log(ctx, AV_LOG_ERROR,

                       "Expected in channel name, got \"%.8s\"\n", arg);

                return AVERROR(EINVAL);

            }

            nb_in_channels[named]++;

            if (nb_in_channels[!named]) {

                av_log(ctx, AV_LOG_ERROR,

                       "Can not mix named and numbered channels\n");

                return AVERROR(EINVAL);

            }

            pan->gain[out_ch_id][in_ch_id] = gain;

            if (!*arg)

                break;

            if (*arg != '+') {

                av_log(ctx, AV_LOG_ERROR, "Syntax error near \"%.8s\"\n", arg);

                return AVERROR(EINVAL);

            }

            arg++;

            skip_spaces(&arg);

        }

    }

    pan->need_renumber = !!nb_in_channels[1];



    av_free(args);

    return 0;

}
