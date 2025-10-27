int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {

    const uint64_t fuzz_tag = FUZZ_TAG;

    FuzzDataBuffer buffer;

    const uint8_t *last = data;

    const uint8_t *end = data + size;

    uint32_t it = 0;



    if (!c)

        c = AVCodecInitialize(FFMPEG_CODEC);  // Done once.



    AVCodecContext* ctx = avcodec_alloc_context3(NULL);

    if (!ctx)

        error("Failed memory allocation");



    ctx->max_pixels = 4096 * 4096; //To reduce false positive OOM and hangs



    int res = avcodec_open2(ctx, c, NULL);

    if (res < 0)

        return res;



    FDBCreate(&buffer);

    int got_frame;

    AVFrame *frame = av_frame_alloc();

    if (!frame)

        error("Failed memory allocation");



    // Read very simple container

    AVPacket avpkt;

    while (data < end && it < maxiteration) {

        // Search for the TAG

        while (data + sizeof(fuzz_tag) < end) {

            if (data[0] == (fuzz_tag & 0xFF) && *(const uint64_t *)(data) == fuzz_tag)

                break;

            data++;

        }

        if (data + sizeof(fuzz_tag) > end)

            data = end;



        FDBPrepare(&buffer, &avpkt, last, data - last);

        data += sizeof(fuzz_tag);

        last = data;



        // Iterate through all data

        while (avpkt.size > 0 && it++ < maxiteration) {

            av_frame_unref(frame);

            int ret = decode_handler(ctx, frame, &got_frame, &avpkt);



            if (it > 20)

                ctx->error_concealment = 0;



            if (ret <= 0 || ret > avpkt.size)

               break;



            avpkt.data += ret;

            avpkt.size -= ret;

        }

    }



    av_init_packet(&avpkt);

    avpkt.data = NULL;

    avpkt.size = 0;



    do {

        got_frame = 0;

        decode_handler(ctx, frame, &got_frame, &avpkt);

    } while (got_frame == 1 && it++ < maxiteration);



    av_frame_free(&frame);

    avcodec_free_context(&ctx);

    av_freep(&ctx);

    FDBDesroy(&buffer);

    return 0;

}