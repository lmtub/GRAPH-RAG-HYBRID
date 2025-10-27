static void do_video_out(AVFormatContext *s,

                         OutputStream *ost,

                         AVFrame *in_picture,

                         float quality)

{

    int ret, format_video_sync;

    AVPacket pkt;

    AVCodecContext *enc = ost->st->codec;

    int nb_frames;

    double sync_ipts, delta;

    double duration = 0;

    int frame_size = 0;

    InputStream *ist = NULL;



    if (ost->source_index >= 0)

        ist = input_streams[ost->source_index];



    if(ist && ist->st->start_time != AV_NOPTS_VALUE && ist->st->first_dts != AV_NOPTS_VALUE && ost->frame_rate.num)

        duration = 1/(av_q2d(ost->frame_rate) * av_q2d(enc->time_base));



    sync_ipts = in_picture->pts;

    delta = sync_ipts - ost->sync_opts + duration;



    /* by default, we output a single frame */

    nb_frames = 1;



    format_video_sync = video_sync_method;

    if (format_video_sync == VSYNC_AUTO)

        format_video_sync = (s->oformat->flags & AVFMT_VARIABLE_FPS) ? ((s->oformat->flags & AVFMT_NOTIMESTAMPS) ? VSYNC_PASSTHROUGH : VSYNC_VFR) : 1;



    switch (format_video_sync) {

    case VSYNC_CFR:

        // FIXME set to 0.5 after we fix some dts/pts bugs like in avidec.c

        if (delta < -1.1)

            nb_frames = 0;

        else if (delta > 1.1)

            nb_frames = lrintf(delta);

        break;

    case VSYNC_VFR:

        if (delta <= -0.6)

            nb_frames = 0;

        else if (delta > 0.6)

            ost->sync_opts = lrint(sync_ipts);

        break;

    case VSYNC_DROP:

    case VSYNC_PASSTHROUGH:

        ost->sync_opts = lrint(sync_ipts);

        break;

    default:

        av_assert0(0);




    nb_frames = FFMIN(nb_frames, ost->max_frames - ost->frame_number);

    if (nb_frames == 0) {


        av_log(NULL, AV_LOG_VERBOSE, "*** drop!\n");


    } else if (nb_frames > 1) {






        nb_frames_dup += nb_frames - 1;

        av_log(NULL, AV_LOG_VERBOSE, "*** %d dup!\n", nb_frames - 1);






duplicate_frame:

    av_init_packet(&pkt);

    pkt.data = NULL;

    pkt.size = 0;



    in_picture->pts = ost->sync_opts;



    if (s->oformat->flags & AVFMT_RAWPICTURE &&

        enc->codec->id == CODEC_ID_RAWVIDEO) {

        /* raw pictures are written as AVPicture structure to

           avoid any copies. We support temporarily the older

           method. */

        enc->coded_frame->interlaced_frame = in_picture->interlaced_frame;

        enc->coded_frame->top_field_first  = in_picture->top_field_first;

        pkt.data   = (uint8_t *)in_picture;

        pkt.size   =  sizeof(AVPicture);

        pkt.pts    = av_rescale_q(in_picture->pts, enc->time_base, ost->st->time_base);

        pkt.flags |= AV_PKT_FLAG_KEY;



        write_frame(s, &pkt, ost);

        video_size += pkt.size;

    } else {

        int got_packet;

        AVFrame big_picture;



        big_picture = *in_picture;

        /* better than nothing: use input picture interlaced

           settings */

        big_picture.interlaced_frame = in_picture->interlaced_frame;

        if (ost->st->codec->flags & (CODEC_FLAG_INTERLACED_DCT|CODEC_FLAG_INTERLACED_ME)) {

            if (ost->top_field_first == -1)

                big_picture.top_field_first = in_picture->top_field_first;

            else

                big_picture.top_field_first = !!ost->top_field_first;




        /* handles same_quant here. This is not correct because it may

           not be a global option */

        big_picture.quality = quality;

        if (!enc->me_threshold)

            big_picture.pict_type = 0;

        if (ost->forced_kf_index < ost->forced_kf_count &&

            big_picture.pts >= ost->forced_kf_pts[ost->forced_kf_index]) {

            big_picture.pict_type = AV_PICTURE_TYPE_I;

            ost->forced_kf_index++;


        update_benchmark(NULL);

        ret = avcodec_encode_video2(enc, &pkt, &big_picture, &got_packet);

        update_benchmark("encode_video %d.%d", ost->file_index, ost->index);

        if (ret < 0) {

            av_log(NULL, AV_LOG_FATAL, "Video encoding failed\n");

            exit_program(1);




        if (got_packet) {

            if (pkt.pts == AV_NOPTS_VALUE && !(enc->codec->capabilities & CODEC_CAP_DELAY))

                pkt.pts = ost->sync_opts;



            if (pkt.pts != AV_NOPTS_VALUE)

                pkt.pts = av_rescale_q(pkt.pts, enc->time_base, ost->st->time_base);

            if (pkt.dts != AV_NOPTS_VALUE)

                pkt.dts = av_rescale_q(pkt.dts, enc->time_base, ost->st->time_base);



            if (debug_ts) {

                av_log(NULL, AV_LOG_INFO, "encoder -> type:video "

                    "pkt_pts:%s pkt_pts_time:%s pkt_dts:%s pkt_dts_time:%s\n",

                    av_ts2str(pkt.pts), av_ts2timestr(pkt.pts, &ost->st->time_base),

                    av_ts2str(pkt.dts), av_ts2timestr(pkt.dts, &ost->st->time_base));




            write_frame(s, &pkt, ost);

            frame_size = pkt.size;

            video_size += pkt.size;

            av_free_packet(&pkt);



            /* if two pass, output log */

            if (ost->logfile && enc->stats_out) {

                fprintf(ost->logfile, "%s", enc->stats_out);




    ost->sync_opts++;

    /*

     * For video, number of frames in == number of packets out.

     * But there may be reordering, so we can't throw away frames on encoder

     * flush, we need to limit them here, before they go into encoder.

     */

    ost->frame_number++;



    if(--nb_frames)

        goto duplicate_frame;



    if (vstats_filename && frame_size)

        do_video_stats(output_files[ost->file_index]->ctx, ost, frame_size);
