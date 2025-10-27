static void do_subtitle_out(AVFormatContext *s,

                            AVOutputStream *ost,

                            AVInputStream *ist,

                            AVSubtitle *sub,

                            int64_t pts)

{

    static uint8_t *subtitle_out = NULL;

    int subtitle_out_max_size = 65536;

    int subtitle_out_size, nb, i;

    AVCodecContext *enc;

    AVPacket pkt;



    if (pts == AV_NOPTS_VALUE) {

        fprintf(stderr, "Subtitle packets must have a pts\n");

        if (exit_on_error)


        return;




    enc = ost->st->codec;



    if (!subtitle_out) {

        subtitle_out = av_malloc(subtitle_out_max_size);




    /* Note: DVB subtitle need one packet to draw them and one other

       packet to clear them */

    /* XXX: signal it in the codec context ? */

    if (enc->codec_id == CODEC_ID_DVB_SUBTITLE)

        nb = 2;

    else

        nb = 1;



    for(i = 0; i < nb; i++) {

        sub->pts = av_rescale_q(pts, ist->st->time_base, AV_TIME_BASE_Q);

        subtitle_out_size = avcodec_encode_subtitle(enc, subtitle_out,

                                                    subtitle_out_max_size, sub);







        av_init_packet(&pkt);

        pkt.stream_index = ost->index;

        pkt.data = subtitle_out;

        pkt.size = subtitle_out_size;

        pkt.pts = av_rescale_q(pts, ist->st->time_base, ost->st->time_base);

        if (enc->codec_id == CODEC_ID_DVB_SUBTITLE) {

            /* XXX: the pts correction is handled here. Maybe handling

               it in the codec would be better */

            if (i == 0)

                pkt.pts += 90 * sub->start_display_time;

            else

                pkt.pts += 90 * sub->end_display_time;


        write_frame(s, &pkt, ost->st->codec, bitstream_filters[ost->file_index][pkt.stream_index]);

