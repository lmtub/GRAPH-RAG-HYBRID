int main (int argc, char **argv)

{

    int ret = 0, got_frame;



    if (argc != 4 && argc != 5) {

        fprintf(stderr, "usage: %s [-refcount=<old|new_norefcount|new_refcount>] "

                "input_file video_output_file audio_output_file\n"

                "API example program to show how to read frames from an input file.\n"

                "This program reads frames from a file, decodes them, and writes decoded\n"

                "video frames to a rawvideo file named video_output_file, and decoded\n"

                "audio frames to a rawaudio file named audio_output_file.\n\n"

                "If the -refcount option is specified, the program use the\n"

                "reference counting frame system which allows keeping a copy of\n"

                "the data for longer than one decode call. If unset, it's using\n"

                "the classic old method.\n"

                "\n", argv[0]);

        exit(1);

    }

    if (argc == 5) {

        const char *mode = argv[1] + strlen("-refcount=");

        if      (!strcmp(mode, "old"))            api_mode = API_MODE_OLD;

        else if (!strcmp(mode, "new_norefcount")) api_mode = API_MODE_NEW_API_NO_REF_COUNT;

        else if (!strcmp(mode, "new_refcount"))   api_mode = API_MODE_NEW_API_REF_COUNT;

        else {

            fprintf(stderr, "unknow mode '%s'\n", mode);

            exit(1);

        }

        argv++;

    }

    src_filename = argv[1];

    video_dst_filename = argv[2];

    audio_dst_filename = argv[3];



    /* register all formats and codecs */

    av_register_all();



    /* open input file, and allocate format context */

    if (avformat_open_input(&fmt_ctx, src_filename, NULL, NULL) < 0) {

        fprintf(stderr, "Could not open source file %s\n", src_filename);

        exit(1);

    }



    /* retrieve stream information */

    if (avformat_find_stream_info(fmt_ctx, NULL) < 0) {

        fprintf(stderr, "Could not find stream information\n");

        exit(1);

    }



    if (open_codec_context(&video_stream_idx, fmt_ctx, AVMEDIA_TYPE_VIDEO) >= 0) {

        video_stream = fmt_ctx->streams[video_stream_idx];

        video_dec_ctx = video_stream->codec;



        video_dst_file = fopen(video_dst_filename, "wb");

        if (!video_dst_file) {

            fprintf(stderr, "Could not open destination file %s\n", video_dst_filename);

            ret = 1;

            goto end;

        }



        /* allocate image where the decoded image will be put */

        ret = av_image_alloc(video_dst_data, video_dst_linesize,

                             video_dec_ctx->width, video_dec_ctx->height,

                             video_dec_ctx->pix_fmt, 1);

        if (ret < 0) {

            fprintf(stderr, "Could not allocate raw video buffer\n");

            goto end;

        }

        video_dst_bufsize = ret;

    }



    if (open_codec_context(&audio_stream_idx, fmt_ctx, AVMEDIA_TYPE_AUDIO) >= 0) {

        audio_stream = fmt_ctx->streams[audio_stream_idx];

        audio_dec_ctx = audio_stream->codec;

        audio_dst_file = fopen(audio_dst_filename, "wb");

        if (!audio_dst_file) {

            fprintf(stderr, "Could not open destination file %s\n", video_dst_filename);

            ret = 1;

            goto end;

        }

    }



    /* dump input information to stderr */

    av_dump_format(fmt_ctx, 0, src_filename, 0);



    if (!audio_stream && !video_stream) {

        fprintf(stderr, "Could not find audio or video stream in the input, aborting\n");

        ret = 1;

        goto end;

    }



    /* When using the new API, you need to use the libavutil/frame.h API, while

     * the classic frame management is available in libavcodec */

    if (api_mode == API_MODE_OLD)

        frame = avcodec_alloc_frame();

    else

        frame = av_frame_alloc();

    if (!frame) {

        fprintf(stderr, "Could not allocate frame\n");

        ret = AVERROR(ENOMEM);

        goto end;

    }



    /* initialize packet, set data to NULL, let the demuxer fill it */

    av_init_packet(&pkt);

    pkt.data = NULL;

    pkt.size = 0;



    if (video_stream)

        printf("Demuxing video from file '%s' into '%s'\n", src_filename, video_dst_filename);

    if (audio_stream)

        printf("Demuxing audio from file '%s' into '%s'\n", src_filename, audio_dst_filename);



    /* read frames from the file */

    while (av_read_frame(fmt_ctx, &pkt) >= 0) {

        AVPacket orig_pkt = pkt;

        do {

            ret = decode_packet(&got_frame, 0);

            if (ret < 0)

                break;

            pkt.data += ret;

            pkt.size -= ret;

        } while (pkt.size > 0);

        av_free_packet(&orig_pkt);

    }



    /* flush cached frames */

    pkt.data = NULL;

    pkt.size = 0;

    do {

        decode_packet(&got_frame, 1);

    } while (got_frame);



    printf("Demuxing succeeded.\n");



    if (video_stream) {

        printf("Play the output video file with the command:\n"

               "ffplay -f rawvideo -pix_fmt %s -video_size %dx%d %s\n",

               av_get_pix_fmt_name(video_dec_ctx->pix_fmt), video_dec_ctx->width, video_dec_ctx->height,

               video_dst_filename);

    }



    if (audio_stream) {

        enum AVSampleFormat sfmt = audio_dec_ctx->sample_fmt;

        int n_channels = audio_dec_ctx->channels;

        const char *fmt;



        if (av_sample_fmt_is_planar(sfmt)) {

            const char *packed = av_get_sample_fmt_name(sfmt);

            printf("Warning: the sample format the decoder produced is planar "

                   "(%s). This example will output the first channel only.\n",

                   packed ? packed : "?");

            sfmt = av_get_packed_sample_fmt(sfmt);

            n_channels = 1;

        }



        if ((ret = get_format_from_sample_fmt(&fmt, sfmt)) < 0)

            goto end;



        printf("Play the output audio file with the command:\n"

               "ffplay -f %s -ac %d -ar %d %s\n",

               fmt, n_channels, audio_dec_ctx->sample_rate,

               audio_dst_filename);

    }



end:

    if (video_dec_ctx)

        avcodec_close(video_dec_ctx);

    if (audio_dec_ctx)

        avcodec_close(audio_dec_ctx);

    avformat_close_input(&fmt_ctx);

    if (video_dst_file)

        fclose(video_dst_file);

    if (audio_dst_file)

        fclose(audio_dst_file);

    if (api_mode == API_MODE_OLD)

        avcodec_free_frame(&frame);

    else

        av_frame_free(&frame);

    av_free(video_dst_data[0]);



    return ret < 0;

}
