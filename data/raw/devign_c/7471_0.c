static int decode_unregistered_user_data(H264SEIUnregistered *h, GetBitContext *gb,

                                         void *logctx, int size)

{

    uint8_t *user_data;

    int e, build, i;



    if (size < 16 || size >= INT_MAX - 16)

        return AVERROR_INVALIDDATA;



    user_data = av_malloc(16 + size + 1);

    if (!user_data)

        return AVERROR(ENOMEM);



    for (i = 0; i < size + 16; i++)

        user_data[i] = get_bits(gb, 8);



    user_data[i] = 0;

    e = sscanf(user_data + 16, "x264 - core %d", &build);

    if (e == 1 && build > 0)

        h->x264_build = build;

    if (e == 1 && build == 1 && !strncmp(user_data+16, "x264 - core 0000", 16))

        h->x264_build = 67;



    if (strlen(user_data + 16) > 0)

        av_log(logctx, AV_LOG_DEBUG, "user data:\"%s\"\n", user_data + 16);



    av_free(user_data);

    return 0;

}
