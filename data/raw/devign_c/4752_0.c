static void print_final_stats(int64_t total_size)

{

    uint64_t video_size = 0, audio_size = 0, extra_size = 0, other_size = 0;

    uint64_t subtitle_size = 0;

    uint64_t data_size = 0;

    float percent = -1.0;

    int i, j;



    for (i = 0; i < nb_output_streams; i++) {

        OutputStream *ost = output_streams[i];

        switch (ost->st->codec->codec_type) {

            case AVMEDIA_TYPE_VIDEO: video_size += ost->data_size; break;

            case AVMEDIA_TYPE_AUDIO: audio_size += ost->data_size; break;

            case AVMEDIA_TYPE_SUBTITLE: subtitle_size += ost->data_size; break;

            default:                 other_size += ost->data_size; break;

        }

        extra_size += ost->st->codec->extradata_size;

        data_size  += ost->data_size;

    }



    if (data_size && total_size >= data_size)

        percent = 100.0 * (total_size - data_size) / data_size;



    av_log(NULL, AV_LOG_INFO, "\n");

    av_log(NULL, AV_LOG_INFO, "video:%1.0fkB audio:%1.0fkB subtitle:%1.0fkB other streams:%1.0fkB global headers:%1.0fkB muxing overhead: ",

           video_size / 1024.0,

           audio_size / 1024.0,

           subtitle_size / 1024.0,

           other_size / 1024.0,

           extra_size / 1024.0);

    if (percent >= 0.0)

        av_log(NULL, AV_LOG_INFO, "%f%%", percent);

    else

        av_log(NULL, AV_LOG_INFO, "unknown");

    av_log(NULL, AV_LOG_INFO, "\n");



    /* print verbose per-stream stats */

    for (i = 0; i < nb_input_files; i++) {

        InputFile *f = input_files[i];

        uint64_t total_packets = 0, total_size = 0;



        av_log(NULL, AV_LOG_VERBOSE, "Input file #%d (%s):\n",

               i, f->ctx->filename);



        for (j = 0; j < f->nb_streams; j++) {

            InputStream *ist = input_streams[f->ist_index + j];

            enum AVMediaType type = ist->st->codec->codec_type;



            total_size    += ist->data_size;

            total_packets += ist->nb_packets;



            av_log(NULL, AV_LOG_VERBOSE, "  Input stream #%d:%d (%s): ",

                   i, j, media_type_string(type));

            av_log(NULL, AV_LOG_VERBOSE, "%"PRIu64" packets read (%"PRIu64" bytes); ",

                   ist->nb_packets, ist->data_size);



            if (ist->decoding_needed) {

                av_log(NULL, AV_LOG_VERBOSE, "%"PRIu64" frames decoded",

                       ist->frames_decoded);

                if (type == AVMEDIA_TYPE_AUDIO)

                    av_log(NULL, AV_LOG_VERBOSE, " (%"PRIu64" samples)", ist->samples_decoded);

                av_log(NULL, AV_LOG_VERBOSE, "; ");

            }



            av_log(NULL, AV_LOG_VERBOSE, "\n");

        }



        av_log(NULL, AV_LOG_VERBOSE, "  Total: %"PRIu64" packets (%"PRIu64" bytes) demuxed\n",

               total_packets, total_size);

    }



    for (i = 0; i < nb_output_files; i++) {

        OutputFile *of = output_files[i];

        uint64_t total_packets = 0, total_size = 0;



        av_log(NULL, AV_LOG_VERBOSE, "Output file #%d (%s):\n",

               i, of->ctx->filename);



        for (j = 0; j < of->ctx->nb_streams; j++) {

            OutputStream *ost = output_streams[of->ost_index + j];

            enum AVMediaType type = ost->st->codec->codec_type;



            total_size    += ost->data_size;

            total_packets += ost->packets_written;



            av_log(NULL, AV_LOG_VERBOSE, "  Output stream #%d:%d (%s): ",

                   i, j, media_type_string(type));

            if (ost->encoding_needed) {

                av_log(NULL, AV_LOG_VERBOSE, "%"PRIu64" frames encoded",

                       ost->frames_encoded);

                if (type == AVMEDIA_TYPE_AUDIO)

                    av_log(NULL, AV_LOG_VERBOSE, " (%"PRIu64" samples)", ost->samples_encoded);

                av_log(NULL, AV_LOG_VERBOSE, "; ");

            }



            av_log(NULL, AV_LOG_VERBOSE, "%"PRIu64" packets muxed (%"PRIu64" bytes); ",

                   ost->packets_written, ost->data_size);



            av_log(NULL, AV_LOG_VERBOSE, "\n");

        }



        av_log(NULL, AV_LOG_VERBOSE, "  Total: %"PRIu64" packets (%"PRIu64" bytes) muxed\n",

               total_packets, total_size);

    }

    if(video_size + data_size + audio_size + subtitle_size + extra_size == 0){

        av_log(NULL, AV_LOG_WARNING, "Output file is empty, nothing was encoded (check -ss / -t / -frames parameters if used)\n");

    }

}
