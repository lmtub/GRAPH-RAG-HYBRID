static void show_packets(AVFormatContext *fmt_ctx)

{

    AVPacket pkt;



    av_init_packet(&pkt);



    while (!av_read_frame(fmt_ctx, &pkt))

        show_packet(fmt_ctx, &pkt);

}
