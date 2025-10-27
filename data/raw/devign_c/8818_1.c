static int avisynth_read_packet_video(AVFormatContext *s, AVPacket *pkt, int discard) {

    AviSynthContext *avs = s->priv_data;

    AVS_VideoFrame *frame;

    unsigned char *dst_p;

    const unsigned char *src_p;

    int n, i, plane, rowsize, planeheight, pitch, bits;

    const char *error;



    if (avs->curr_frame >= avs->vi->num_frames)

        return AVERROR_EOF;



    // This must happen even if the stream is discarded to prevent desync.

    n = avs->curr_frame++;

    if (discard)

        return 0;



    pkt->pts = n;

    pkt->dts = n;

    pkt->duration = 1;



    // Define the bpp values for the new AviSynth 2.6 colorspaces

    if (avs_is_yv24(avs->vi)) {

        bits = 24;

    } else if (avs_is_yv16(avs->vi)) {

        bits = 16;

    } else if (avs_is_yv411(avs->vi)) {

        bits = 12;

    } else if (avs_is_y8(avs->vi)) {

        bits = 8;


        bits = avs_bits_per_pixel(avs->vi);




    // Without cast to int64_t, calculation overflows at about 9k x 9k resolution.

    pkt->size = (((int64_t)avs->vi->width * (int64_t)avs->vi->height) * bits) / 8;

    if (!pkt->size)


    pkt->data = av_malloc(pkt->size);

    if (!pkt->data)




    frame = avs_library->avs_get_frame(avs->clip, n);

    error = avs_library->avs_clip_get_error(avs->clip);

    if (error) {

        av_log(s, AV_LOG_ERROR, "%s\n", error);







    dst_p = pkt->data;

    for (i = 0; i < avs->n_planes; i++) {

        plane = avs->planes[i];

        src_p = avs_get_read_ptr_p(frame, plane);

        rowsize = avs_get_row_size_p(frame, plane);

        planeheight = avs_get_height_p(frame, plane);

        pitch = avs_get_pitch_p(frame, plane);



        // Flip RGB video.

        if (avs_is_rgb24(avs->vi) || avs_is_rgb(avs->vi)) {

            src_p = src_p + (planeheight - 1) * pitch;

            pitch = -pitch;
























        dst_p += rowsize * planeheight;




    avs_library->avs_release_video_frame(frame);

    return 0;
