static int check_init_output_file(OutputFile *of, int file_index)

{

    int ret, i;



    for (i = 0; i < of->ctx->nb_streams; i++) {

        OutputStream *ost = output_streams[of->ost_index + i];

        if (!ost->initialized)

            return 0;

    }



    of->ctx->interrupt_callback = int_cb;



    ret = avformat_write_header(of->ctx, &of->opts);

    if (ret < 0) {

        av_log(NULL, AV_LOG_ERROR,

               "Could not write header for output file #%d "

               "(incorrect codec parameters ?): %s\n",

               file_index, av_err2str(ret));

        return ret;

    }

    //assert_avoptions(of->opts);

    of->header_written = 1;



    av_dump_format(of->ctx, file_index, of->ctx->filename, 1);



    if (sdp_filename || want_sdp)

        print_sdp();



    /* flush the muxing queues */

    for (i = 0; i < of->ctx->nb_streams; i++) {

        OutputStream *ost = output_streams[of->ost_index + i];



        /* try to improve muxing time_base (only possible if nothing has been written yet) */

        if (!av_fifo_size(ost->muxing_queue))

            ost->mux_timebase = ost->st->time_base;



        while (av_fifo_size(ost->muxing_queue)) {

            AVPacket pkt;

            av_fifo_generic_read(ost->muxing_queue, &pkt, sizeof(pkt), NULL);

            write_packet(of, &pkt, ost);

        }

    }



    return 0;

}
