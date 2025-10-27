static void show_packets(WriterContext *w, AVFormatContext *fmt_ctx)

{

    AVPacket pkt;

    AVFrame frame;

    int i = 0;



    av_init_packet(&pkt);



    while (!av_read_frame(fmt_ctx, &pkt)) {

        if (do_show_packets)

            show_packet(w, fmt_ctx, &pkt, i++);

        if (do_show_frames &&

            get_video_frame(fmt_ctx, &frame, &pkt)) {

            show_frame(w, &frame, fmt_ctx->streams[pkt.stream_index]);

            av_destruct_packet(&pkt);

        }

    }

    av_init_packet(&pkt);

    pkt.data = NULL;

    pkt.size = 0;

    //Flush remaining frames that are cached in the decoder

    for (i = 0; i < fmt_ctx->nb_streams; i++) {

        pkt.stream_index = i;

        while (get_video_frame(fmt_ctx, &frame, &pkt))

            show_frame(w, &frame, fmt_ctx->streams[pkt.stream_index]);

    }

}
