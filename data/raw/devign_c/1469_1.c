static int probe_file(WriterContext *wctx, const char *filename)

{

    AVFormatContext *fmt_ctx;

    int ret, i;

    int section_id;



    do_read_frames = do_show_frames || do_count_frames;

    do_read_packets = do_show_packets || do_count_packets;



    ret = open_input_file(&fmt_ctx, filename);

    if (ret < 0)

        return ret;



#define CHECK_END if (ret < 0) goto end



    nb_streams_frames  = av_calloc(fmt_ctx->nb_streams, sizeof(*nb_streams_frames));

    nb_streams_packets = av_calloc(fmt_ctx->nb_streams, sizeof(*nb_streams_packets));

    selected_streams   = av_calloc(fmt_ctx->nb_streams, sizeof(*selected_streams));



    for (i = 0; i < fmt_ctx->nb_streams; i++) {

        if (stream_specifier) {

            ret = avformat_match_stream_specifier(fmt_ctx,

                                                  fmt_ctx->streams[i],

                                                  stream_specifier);

            CHECK_END;

            else

                selected_streams[i] = ret;

            ret = 0;

        } else {

            selected_streams[i] = 1;

        }

    }



    if (do_read_frames || do_read_packets) {

        if (do_show_frames && do_show_packets &&

            wctx->writer->flags & WRITER_FLAG_PUT_PACKETS_AND_FRAMES_IN_SAME_CHAPTER)

            section_id = SECTION_ID_PACKETS_AND_FRAMES;

        else if (do_show_packets && !do_show_frames)

            section_id = SECTION_ID_PACKETS;

        else // (!do_show_packets && do_show_frames)

            section_id = SECTION_ID_FRAMES;

        if (do_show_frames || do_show_packets)

            writer_print_section_header(wctx, section_id);

        ret = read_packets(wctx, fmt_ctx);

        if (do_show_frames || do_show_packets)

            writer_print_section_footer(wctx);

        CHECK_END;

    }



    if (do_show_programs) {

        ret = show_programs(wctx, fmt_ctx);

        CHECK_END;

    }



    if (do_show_streams) {

        ret = show_streams(wctx, fmt_ctx);

        CHECK_END;

    }

    if (do_show_chapters) {

        ret = show_chapters(wctx, fmt_ctx);

        CHECK_END;

    }

    if (do_show_format) {

        ret = show_format(wctx, fmt_ctx);

        CHECK_END;

    }



end:

    close_input_file(&fmt_ctx);

    av_freep(&nb_streams_frames);

    av_freep(&nb_streams_packets);

    av_freep(&selected_streams);



    return ret;

}
