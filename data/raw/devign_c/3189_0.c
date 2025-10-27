static void peak_write_chunk(AVFormatContext *s)

{

    WAVMuxContext *wav = s->priv_data;

    AVIOContext *pb = s->pb;

    AVCodecContext *enc = s->streams[0]->codec;

    int64_t peak = ff_start_tag(s->pb, "levl");

    int64_t now0;

    time_t now_secs;

    char timestamp[28];



    /* Peak frame of incomplete block at end */

    if (wav->peak_block_pos)

        peak_write_frame(s);



    memset(timestamp, 0, sizeof(timestamp));

    if (!(s->flags & AVFMT_FLAG_BITEXACT)) {

        struct tm tmpbuf;

        av_log(s, AV_LOG_INFO, "Writing local time and date to Peak Envelope Chunk\n");

        now0 = av_gettime();

        now_secs = now0 / 1000000;

        strftime(timestamp, sizeof(timestamp), "%Y:%m:%d:%H:%M:%S:", localtime_r(&now_secs, &tmpbuf));

        av_strlcatf(timestamp, sizeof(timestamp), "%03d", (int)((now0 / 1000) % 1000));

    }



    avio_wl32(pb, 1);                           /* version */

    avio_wl32(pb, wav->peak_format);            /* 8 or 16 bit */

    avio_wl32(pb, wav->peak_ppv);               /* positive and negative */

    avio_wl32(pb, wav->peak_block_size);        /* frames per value */

    avio_wl32(pb, enc->channels);               /* number of channels */

    avio_wl32(pb, wav->peak_num_frames);        /* number of peak frames */

    avio_wl32(pb, wav->peak_pos_pop);           /* audio sample frame index */

    avio_wl32(pb, 128);                         /* equal to size of header */

    avio_write(pb, timestamp, 28);              /* ASCII time stamp */

    ffio_fill(pb, 0, 60);



    avio_write(pb, wav->peak_output, wav->peak_outbuf_bytes);



    ff_end_tag(pb, peak);



    if (!wav->data)

        wav->data = peak;

}
