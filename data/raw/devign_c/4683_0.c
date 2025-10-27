static void avconv_cleanup(int ret)

{

    int i, j;



    for (i = 0; i < nb_filtergraphs; i++) {

        FilterGraph *fg = filtergraphs[i];

        avfilter_graph_free(&fg->graph);

        for (j = 0; j < fg->nb_inputs; j++) {

            while (av_fifo_size(fg->inputs[j]->frame_queue)) {

                AVFrame *frame;

                av_fifo_generic_read(fg->inputs[j]->frame_queue, &frame,

                                     sizeof(frame), NULL);

                av_frame_free(&frame);

            }

            av_fifo_free(fg->inputs[j]->frame_queue);

            av_buffer_unref(&fg->inputs[j]->hw_frames_ctx);

            av_freep(&fg->inputs[j]->name);

            av_freep(&fg->inputs[j]);

        }

        av_freep(&fg->inputs);

        for (j = 0; j < fg->nb_outputs; j++) {

            av_freep(&fg->outputs[j]->name);

            av_freep(&fg->outputs[j]->formats);

            av_freep(&fg->outputs[j]->channel_layouts);

            av_freep(&fg->outputs[j]->sample_rates);

            av_freep(&fg->outputs[j]);

        }

        av_freep(&fg->outputs);

        av_freep(&fg->graph_desc);



        av_freep(&filtergraphs[i]);

    }

    av_freep(&filtergraphs);



    /* close files */

    for (i = 0; i < nb_output_files; i++) {

        OutputFile *of = output_files[i];

        AVFormatContext *s = of->ctx;

        if (s && s->oformat && !(s->oformat->flags & AVFMT_NOFILE) && s->pb)

            avio_close(s->pb);

        avformat_free_context(s);

        av_dict_free(&of->opts);



        av_freep(&output_files[i]);

    }

    for (i = 0; i < nb_output_streams; i++) {

        OutputStream *ost = output_streams[i];



        for (j = 0; j < ost->nb_bitstream_filters; j++)

            av_bsf_free(&ost->bsf_ctx[j]);

        av_freep(&ost->bsf_ctx);

        av_freep(&ost->bitstream_filters);



        av_frame_free(&ost->filtered_frame);



        av_parser_close(ost->parser);

        avcodec_free_context(&ost->parser_avctx);



        av_freep(&ost->forced_keyframes);

        av_freep(&ost->avfilter);

        av_freep(&ost->logfile_prefix);



        avcodec_free_context(&ost->enc_ctx);



        while (av_fifo_size(ost->muxing_queue)) {

            AVPacket pkt;

            av_fifo_generic_read(ost->muxing_queue, &pkt, sizeof(pkt), NULL);

            av_packet_unref(&pkt);

        }

        av_fifo_free(ost->muxing_queue);



        av_freep(&output_streams[i]);

    }

    for (i = 0; i < nb_input_files; i++) {

        avformat_close_input(&input_files[i]->ctx);

        av_freep(&input_files[i]);

    }

    for (i = 0; i < nb_input_streams; i++) {

        InputStream *ist = input_streams[i];



        av_frame_free(&ist->decoded_frame);

        av_frame_free(&ist->filter_frame);

        av_dict_free(&ist->decoder_opts);

        av_freep(&ist->filters);

        av_freep(&ist->hwaccel_device);



        avcodec_free_context(&ist->dec_ctx);



        av_freep(&input_streams[i]);

    }



    if (vstats_file)

        fclose(vstats_file);

    av_free(vstats_filename);



    av_freep(&input_streams);

    av_freep(&input_files);

    av_freep(&output_streams);

    av_freep(&output_files);



    uninit_opts();



    avformat_network_deinit();



    if (received_sigterm) {

        av_log(NULL, AV_LOG_INFO, "Received signal %d: terminating.\n",

               (int) received_sigterm);

        exit (255);

    }

}
