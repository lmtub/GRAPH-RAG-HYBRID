int ff_mov_init_hinting(AVFormatContext *s, int index, int src_index)

{

    MOVMuxContext *mov  = s->priv_data;

    MOVTrack *track     = &mov->tracks[index];

    MOVTrack *src_track = &mov->tracks[src_index];

    AVStream *src_st    = s->streams[src_index];

    int ret = AVERROR(ENOMEM);

    AVOutputFormat *rtp_format = av_guess_format("rtp", NULL, NULL);



    track->tag = MKTAG('r','t','p',' ');

    track->src_track = src_index;



    if (!rtp_format) {

        ret = AVERROR(ENOENT);

        goto fail;

    }



    track->enc = avcodec_alloc_context();

    if (!track->enc)

        goto fail;

    track->enc->codec_type = AVMEDIA_TYPE_DATA;

    track->enc->codec_tag  = track->tag;



    track->rtp_ctx = avformat_alloc_context();

    if (!track->rtp_ctx)

        goto fail;

    track->rtp_ctx->oformat = rtp_format;

    if (!av_new_stream(track->rtp_ctx, 0))

        goto fail;



    /* Copy stream parameters */

    track->rtp_ctx->streams[0]->sample_aspect_ratio =

                        src_st->sample_aspect_ratio;



    /* Remove the allocated codec context, link to the original one

     * instead, to give the rtp muxer access to codec parameters. */

    av_free(track->rtp_ctx->streams[0]->codec);

    track->rtp_ctx->streams[0]->codec = src_st->codec;



    if ((ret = url_open_dyn_packet_buf(&track->rtp_ctx->pb,

                                       RTP_MAX_PACKET_SIZE)) < 0)

        goto fail;

    ret = av_write_header(track->rtp_ctx);

    if (ret)

        goto fail;



    /* Copy the RTP AVStream timebase back to the hint AVStream */

    track->timescale = track->rtp_ctx->streams[0]->time_base.den;



    /* Mark the hinted track that packets written to it should be

     * sent to this track for hinting. */

    src_track->hint_track = index;

    return 0;

fail:

    av_log(s, AV_LOG_WARNING,

           "Unable to initialize hinting of stream %d\n", src_index);

    if (track->rtp_ctx && track->rtp_ctx->pb) {

        uint8_t *buf;

        url_close_dyn_buf(track->rtp_ctx->pb, &buf);

        av_free(buf);

    }

    if (track->rtp_ctx && track->rtp_ctx->streams[0]) {

        av_metadata_free(&track->rtp_ctx->streams[0]->metadata);


        av_free(track->rtp_ctx->streams[0]);

    }

    if (track->rtp_ctx) {

        av_metadata_free(&track->rtp_ctx->metadata);

        av_free(track->rtp_ctx->priv_data);

        av_freep(&track->rtp_ctx);

    }

    av_freep(&track->enc);

    /* Set a default timescale, to avoid crashes in dump_format */

    track->timescale = 90000;

    return ret;

}