static int movie_push_frame(AVFilterContext *ctx, unsigned out_id)

{

    MovieContext *movie = ctx->priv;

    AVPacket *pkt = &movie->pkt;

    enum AVMediaType frame_type;

    MovieStream *st;

    int ret, got_frame = 0, pkt_out_id;

    AVFilterLink *outlink;



    if (!pkt->size) {

        if (movie->eof) {

            if (movie->st[out_id].done) {

                if (movie->loop_count != 1) {

                    ret = rewind_file(ctx);

                    if (ret < 0)

                        return ret;

                    movie->loop_count -= movie->loop_count > 1;

                    av_log(ctx, AV_LOG_VERBOSE, "Stream finished, looping.\n");

                    return 0; /* retry */

                }

                return AVERROR_EOF;

            }

            pkt->stream_index = movie->st[out_id].st->index;

            /* packet is already ready for flushing */

        } else {

            ret = av_read_frame(movie->format_ctx, &movie->pkt0);

            if (ret < 0) {

                av_init_packet(&movie->pkt0); /* ready for flushing */

                *pkt = movie->pkt0;

                if (ret == AVERROR_EOF) {

                    movie->eof = 1;

                    return 0; /* start flushing */

                }

                return ret;

            }

            *pkt = movie->pkt0;

        }

    }



    pkt_out_id = pkt->stream_index > movie->max_stream_index ? -1 :

                 movie->out_index[pkt->stream_index];

    if (pkt_out_id < 0) {

        av_free_packet(&movie->pkt0);

        pkt->size = 0; /* ready for next run */

        pkt->data = NULL;

        return 0;

    }

    st = &movie->st[pkt_out_id];

    outlink = ctx->outputs[pkt_out_id];



    movie->frame = av_frame_alloc();

    if (!movie->frame)

        return AVERROR(ENOMEM);



    frame_type = st->st->codec->codec_type;

    switch (frame_type) {

    case AVMEDIA_TYPE_VIDEO:

        ret = avcodec_decode_video2(st->st->codec, movie->frame, &got_frame, pkt);

        break;

    case AVMEDIA_TYPE_AUDIO:

        ret = avcodec_decode_audio4(st->st->codec, movie->frame, &got_frame, pkt);

        break;

    default:

        ret = AVERROR(ENOSYS);

        break;

    }

    if (ret < 0) {

        av_log(ctx, AV_LOG_WARNING, "Decode error: %s\n", av_err2str(ret));

        av_frame_free(&movie->frame);

        av_free_packet(&movie->pkt0);

        movie->pkt.size = 0;

        movie->pkt.data = NULL;

        return 0;

    }

    if (!ret || st->st->codec->codec_type == AVMEDIA_TYPE_VIDEO)

        ret = pkt->size;



    pkt->data += ret;

    pkt->size -= ret;

    if (pkt->size <= 0) {

        av_free_packet(&movie->pkt0);

        pkt->size = 0; /* ready for next run */

        pkt->data = NULL;

    }

    if (!got_frame) {

        if (!ret)

            st->done = 1;

        av_frame_free(&movie->frame);

        return 0;

    }



    movie->frame->pts = av_frame_get_best_effort_timestamp(movie->frame);

    av_dlog(ctx, "movie_push_frame(): file:'%s' %s\n", movie->file_name,

            describe_frame_to_str((char[1024]){0}, 1024, movie->frame, frame_type, outlink));



    if (st->st->codec->codec_type == AVMEDIA_TYPE_VIDEO) {

        if (movie->frame->format != outlink->format) {

            av_log(ctx, AV_LOG_ERROR, "Format changed %s -> %s, discarding frame\n",

                av_get_pix_fmt_name(outlink->format),

                av_get_pix_fmt_name(movie->frame->format)

                );

            av_frame_free(&movie->frame);

            return 0;

        }

    }

    ret = ff_filter_frame(outlink, movie->frame);

    movie->frame = NULL;



    if (ret < 0)

        return ret;

    return pkt_out_id == out_id;

}
