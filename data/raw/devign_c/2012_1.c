static int vp9_superframe_filter(AVBSFContext *ctx, AVPacket *out)

{

    BitstreamContext bc;

    VP9BSFContext *s = ctx->priv_data;

    AVPacket *in;

    int res, invisible, profile, marker, uses_superframe_syntax = 0, n;



    res = ff_bsf_get_packet(ctx, &in);

    if (res < 0)

        return res;



    marker = in->data[in->size - 1];

    if ((marker & 0xe0) == 0xc0) {

        int nbytes = 1 + ((marker >> 3) & 0x3);

        int n_frames = 1 + (marker & 0x7), idx_sz = 2 + n_frames * nbytes;



        uses_superframe_syntax = in->size >= idx_sz && in->data[in->size - idx_sz] == marker;

    }



    res = bitstream_init8(&bc, in->data, in->size);

    if (res < 0)

        goto done;



    bitstream_read(&bc, 2); // frame marker

    profile  = bitstream_read(&bc, 1);

    profile |= bitstream_read(&bc, 1) << 1;

    if (profile == 3)

        profile += bitstream_read(&bc, 1);



    if (bitstream_read(&bc, 1)) {

        invisible = 0;

    } else {

        bitstream_read(&bc, 1); // keyframe

        invisible = !bitstream_read(&bc, 1);

    }



    if (uses_superframe_syntax && s->n_cache > 0) {

        av_log(ctx, AV_LOG_ERROR,

               "Mixing of superframe syntax and naked VP9 frames not supported");

        res = AVERROR(ENOSYS);

        goto done;

    } else if ((!invisible || uses_superframe_syntax) && !s->n_cache) {

        // passthrough

        av_packet_move_ref(out, in);

        goto done;

    } else if (s->n_cache + 1 >= MAX_CACHE) {

        av_log(ctx, AV_LOG_ERROR,

               "Too many invisible frames");

        res = AVERROR_INVALIDDATA;

        goto done;

    }



    s->cache[s->n_cache++] = in;

    in                     = NULL;

    if (invisible) {

        res = AVERROR(EAGAIN);

        goto done;

    }

    av_assert0(s->n_cache > 0);



    // build superframe

    if ((res = merge_superframe(s->cache, s->n_cache, out)) < 0)

        goto done;



    res = av_packet_copy_props(out, s->cache[s->n_cache - 1]);

    if (res < 0)

        goto done;



    for (n = 0; n < s->n_cache; n++)

        av_packet_free(&s->cache[n]);

    s->n_cache = 0;



done:

    if (res < 0)

        av_packet_unref(out);

    av_packet_free(&in);

    return res;

}
