int configure_filtergraph(FilterGraph *fg)

{

    AVFilterInOut *inputs, *outputs, *cur;

    int ret, i, simple = !fg->graph_desc;

    const char *graph_desc = simple ? fg->outputs[0]->ost->avfilter :

                                      fg->graph_desc;



    avfilter_graph_free(&fg->graph);

    if (!(fg->graph = avfilter_graph_alloc()))

        return AVERROR(ENOMEM);



    if (simple) {

        OutputStream *ost = fg->outputs[0]->ost;

        char args[512];

        AVDictionaryEntry *e = NULL;



        args[0] = 0;

        while ((e = av_dict_get(ost->sws_dict, "", e,

                                AV_DICT_IGNORE_SUFFIX))) {

            av_strlcatf(args, sizeof(args), "%s=%s:", e->key, e->value);

        }

        if (strlen(args))

            args[strlen(args)-1] = 0;

        fg->graph->scale_sws_opts = av_strdup(args);



        args[0] = 0;

        while ((e = av_dict_get(ost->swr_opts, "", e,

                                AV_DICT_IGNORE_SUFFIX))) {

            av_strlcatf(args, sizeof(args), "%s=%s:", e->key, e->value);

        }

        if (strlen(args))

            args[strlen(args)-1] = 0;

        av_opt_set(fg->graph, "aresample_swr_opts", args, 0);



        args[0] = '\0';

        while ((e = av_dict_get(fg->outputs[0]->ost->resample_opts, "", e,

                                AV_DICT_IGNORE_SUFFIX))) {

            av_strlcatf(args, sizeof(args), "%s=%s:", e->key, e->value);

        }

        if (strlen(args))

            args[strlen(args) - 1] = '\0';

        fg->graph->resample_lavr_opts = av_strdup(args);



        e = av_dict_get(ost->encoder_opts, "threads", NULL, 0);

        if (e)

            av_opt_set(fg->graph, "threads", e->value, 0);

    }



    if ((ret = avfilter_graph_parse2(fg->graph, graph_desc, &inputs, &outputs)) < 0)

        return ret;



    if (simple && (!inputs || inputs->next || !outputs || outputs->next)) {

        const char *num_inputs;

        const char *num_outputs;

        if (!outputs) {

            num_outputs = "0";

        } else if (outputs->next) {

            num_outputs = ">1";

        } else {

            num_outputs = "1";

        }

        if (!inputs) {

            num_inputs = "0";

        } else if (inputs->next) {

            num_inputs = ">1";

        } else {

            num_inputs = "1";

        }

        av_log(NULL, AV_LOG_ERROR, "Simple filtergraph '%s' was expected "

               "to have exactly 1 input and 1 output."

               " However, it had %s input(s) and %s output(s)."

               " Please adjust, or use a complex filtergraph (-filter_complex) instead.\n",

               graph_desc, num_inputs, num_outputs);

        return AVERROR(EINVAL);

    }



    for (cur = inputs, i = 0; cur; cur = cur->next, i++)

        if ((ret = configure_input_filter(fg, fg->inputs[i], cur)) < 0) {

            avfilter_inout_free(&inputs);

            avfilter_inout_free(&outputs);

            return ret;

        }

    avfilter_inout_free(&inputs);



    for (cur = outputs, i = 0; cur; cur = cur->next, i++)

        configure_output_filter(fg, fg->outputs[i], cur);

    avfilter_inout_free(&outputs);



    if ((ret = avfilter_graph_config(fg->graph, NULL)) < 0)

        return ret;



    fg->reconfiguration = 1;



    for (i = 0; i < fg->nb_outputs; i++) {

        OutputStream *ost = fg->outputs[i]->ost;

        if (!ost->enc) {

            /* identical to the same check in ffmpeg.c, needed because

               complex filter graphs are initialized earlier */

            av_log(NULL, AV_LOG_ERROR, "Encoder (codec %s) not found for output stream #%d:%d\n",

                     avcodec_get_name(ost->st->codec->codec_id), ost->file_index, ost->index);

            return AVERROR(EINVAL);

        }

        if (ost &&

            ost->enc->type == AVMEDIA_TYPE_AUDIO &&

            !(ost->enc->capabilities & AV_CODEC_CAP_VARIABLE_FRAME_SIZE))

            av_buffersink_set_frame_size(ost->filter->filter,

                                         ost->enc_ctx->frame_size);

    }



    return 0;

}
