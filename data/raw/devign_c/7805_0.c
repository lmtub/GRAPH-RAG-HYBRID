static int seek_test(const char *input_filename, const char *start, const char *end)

{

    AVCodec *codec = NULL;

    AVCodecContext *ctx= NULL;

    AVCodecParameters *origin_par = NULL;

    AVFrame *fr = NULL;

    AVFormatContext *fmt_ctx = NULL;

    int video_stream;

    int result;

    int i, j;

    long int start_ts, end_ts;



    size_of_array = 0;

    number_of_elements = 0;

    crc_array = pts_array = NULL;



    result = avformat_open_input(&fmt_ctx, input_filename, NULL, NULL);

    if (result < 0) {

        av_log(NULL, AV_LOG_ERROR, "Can't open file\n");

        return result;

    }



    result = avformat_find_stream_info(fmt_ctx, NULL);

    if (result < 0) {

        av_log(NULL, AV_LOG_ERROR, "Can't get stream info\n");

        return result;

    }



    start_ts = read_seek_range(start);

    end_ts = read_seek_range(end);

    if ((start_ts < 0) || (end_ts < 0))

        return -1;



    //TODO: add ability to work with audio format

    video_stream = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);

    if (video_stream < 0) {

      av_log(NULL, AV_LOG_ERROR, "Can't find video stream in input file\n");

      return -1;

    }



    origin_par = fmt_ctx->streams[video_stream]->codecpar;



    codec = avcodec_find_decoder(origin_par->codec_id);

    if (!codec) {

        av_log(NULL, AV_LOG_ERROR, "Can't find decoder\n");

        return -1;

    }



    ctx = avcodec_alloc_context3(codec);

    if (!ctx) {

        av_log(NULL, AV_LOG_ERROR, "Can't allocate decoder context\n");

        return AVERROR(ENOMEM);

    }



    result = avcodec_parameters_to_context(ctx, origin_par);

    if (result) {

        av_log(NULL, AV_LOG_ERROR, "Can't copy decoder context\n");

        return result;

    }



    result = avcodec_open2(ctx, codec, NULL);

    if (result < 0) {

        av_log(ctx, AV_LOG_ERROR, "Can't open decoder\n");

        return result;

    }



    fr = av_frame_alloc();

    if (!fr) {

        av_log(NULL, AV_LOG_ERROR, "Can't allocate frame\n");

        return AVERROR(ENOMEM);

    }



    result = compute_crc_of_packets(fmt_ctx, video_stream, ctx, fr, 0, 0, 1);

    if (result != 0)

        return -1;



    for (i = start_ts; i < end_ts; i += 100) {

        for (j = i + 100; j < end_ts; j += 100)

        result = compute_crc_of_packets(fmt_ctx, video_stream, ctx, fr, i, j, 0);

        if (result != 0)

            return -1;

    }



    av_freep(&crc_array);

    av_freep(&pts_array);

    av_frame_free(&fr);

    avcodec_close(ctx);

    avformat_close_input(&fmt_ctx);

    avcodec_free_context(&ctx);

    return 0;

}
