static void flush_encoders(void)

{

    int i, ret;



    for (i = 0; i < nb_output_streams; i++) {

        OutputStream   *ost = output_streams[i];

        AVCodecContext *enc = ost->enc_ctx;

        AVFormatContext *os = output_files[ost->file_index]->ctx;

        int stop_encoding = 0;



        if (!ost->encoding_needed)

            continue;



        if (enc->codec_type == AVMEDIA_TYPE_AUDIO && enc->frame_size <= 1)

            continue;



        if (enc->codec_type != AVMEDIA_TYPE_VIDEO && enc->codec_type != AVMEDIA_TYPE_AUDIO)

            continue;



        avcodec_send_frame(enc, NULL);



        for (;;) {

            const char *desc = NULL;



            switch (enc->codec_type) {

            case AVMEDIA_TYPE_AUDIO:

                desc   = "Audio";

                break;

            case AVMEDIA_TYPE_VIDEO:

                desc   = "Video";

                break;

            default:

                av_assert0(0);

            }



            if (1) {

                AVPacket pkt;

                av_init_packet(&pkt);

                pkt.data = NULL;

                pkt.size = 0;



                ret = avcodec_receive_packet(enc, &pkt);

                if (ret < 0 && ret != AVERROR_EOF) {

                    av_log(NULL, AV_LOG_FATAL, "%s encoding failed\n", desc);

                    exit_program(1);

                }

                if (ost->logfile && enc->stats_out) {

                    fprintf(ost->logfile, "%s", enc->stats_out);

                }

                if (ret == AVERROR_EOF) {

                    stop_encoding = 1;

                    break;

                }

                av_packet_rescale_ts(&pkt, enc->time_base, ost->st->time_base);

                output_packet(os, &pkt, ost);

            }



            if (stop_encoding)

                break;

        }

    }

}
