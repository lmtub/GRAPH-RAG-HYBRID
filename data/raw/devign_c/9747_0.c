int av_asrc_buffer_add_audio_buffer_ref(AVFilterContext *ctx,

                                        AVFilterBufferRef *samplesref,

                                        int av_unused flags)

{

    BufferSourceContext *abuffer = ctx->priv;

    AVFilterLink *link;

    int ret, logged = 0;



    if (av_fifo_space(abuffer->fifo) < sizeof(samplesref)) {

        av_log(ctx, AV_LOG_ERROR,

               "Buffering limit reached. Please consume some available frames "

               "before adding new ones.\n");

        return AVERROR(EINVAL);

    }



    // Normalize input



    link = ctx->outputs[0];

    if (samplesref->audio->sample_rate != link->sample_rate) {



        log_input_change(ctx, link, samplesref);

        logged = 1;



        abuffer->sample_rate = samplesref->audio->sample_rate;



        if (!abuffer->aresample) {

            ret = insert_filter(abuffer, link, &abuffer->aresample, "aresample");

            if (ret < 0) return ret;

        } else {

            link = abuffer->aresample->outputs[0];

            if (samplesref->audio->sample_rate == link->sample_rate)

                remove_filter(&abuffer->aresample);

            else

                if ((ret = reconfigure_filter(abuffer, abuffer->aresample)) < 0)

                    return ret;

        }

    }



    link = ctx->outputs[0];

    if (samplesref->format                != link->format         ||

        samplesref->audio->channel_layout != link->channel_layout ||

        samplesref->audio->planar         != link->planar) {



        if (!logged) log_input_change(ctx, link, samplesref);



        abuffer->sample_format  = samplesref->format;

        abuffer->channel_layout = samplesref->audio->channel_layout;

        abuffer->packing_format = samplesref->audio->planar;



        if (!abuffer->aconvert) {

            ret = insert_filter(abuffer, link, &abuffer->aconvert, "aconvert");

            if (ret < 0) return ret;

        } else {

            link = abuffer->aconvert->outputs[0];

            if (samplesref->format                == link->format         &&

                samplesref->audio->channel_layout == link->channel_layout &&

                samplesref->audio->planar         == link->planar

               )

                remove_filter(&abuffer->aconvert);

            else

                if ((ret = reconfigure_filter(abuffer, abuffer->aconvert)) < 0)

                    return ret;

        }

    }



    if (sizeof(samplesref) != av_fifo_generic_write(abuffer->fifo, &samplesref,

                                                    sizeof(samplesref), NULL)) {

        av_log(ctx, AV_LOG_ERROR, "Error while writing to FIFO\n");

        return AVERROR(EINVAL);

    }



    return 0;

}
