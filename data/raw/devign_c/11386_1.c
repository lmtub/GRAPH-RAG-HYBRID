int main (int argc, char **argv)

{

    int ret = 0, got_frame;



    if (argc != 4) {

        fprintf(stderr, "usage: %s input_file video_output_file audio_output_file\n"

                "API example program to show how to read frames from an input file.\n"

                "This program reads frames from a file, decodes them, and writes decoded\n"

                "video frames to a rawvideo file named video_output_file, and decoded\n"

                "audio frames to a rawaudio file named audio_output_file.\n"

                "\n", argv[0]);

        exit(1);

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

        int nb_planes;



        audio_stream = fmt_ctx->streams[audio_stream_idx];

        audio_dec_ctx = audio_stream->codec;

        audio_dst_file = fopen(audio_dst_filename, "wb");

        if (!audio_dst_file) {

            fprintf(stderr, "Could not open destination file %s\n", video_dst_filename);

            ret = 1;

            goto end;

        }



        nb_planes = av_sample_fmt_is_planar(audio_dec_ctx->sample_fmt) ?

            audio_dec_ctx->channels : 1;

        audio_dst_data = av_mallocz(sizeof(uint8_t *) * nb_planes);

        if (!audio_dst_data) {

            fprintf(stderr, "Could not allocate audio data buffers\n");

            ret = AVERROR(ENOMEM);

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



    frame = avcodec_alloc_frame();

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

    while (av_read_frame(fmt_ctx, &pkt) >= 0)

        decode_packet(&got_frame, 0);



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

        const char *fmt;



        if ((ret = get_format_from_sample_fmt(&fmt, audio_dec_ctx->sample_fmt)) < 0)

            goto end;

        printf("Play the output audio file with the command:\n"

               "ffplay -f %s -ac %d -ar %d %s\n",

               fmt, audio_dec_ctx->channels, audio_dec_ctx->sample_rate,

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

    av_free(frame);

    av_free(video_dst_data[0]);

    av_free(audio_dst_data);



    return ret < 0;

}
