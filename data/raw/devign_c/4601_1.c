static int apng_read_header(AVFormatContext *s)

{

    APNGDemuxContext *ctx = s->priv_data;

    AVIOContext *pb = s->pb;

    uint32_t len, tag;

    AVStream *st;

    int acTL_found = 0;

    int64_t ret = AVERROR_INVALIDDATA;



    /* verify PNGSIG */

    if (avio_rb64(pb) != PNGSIG)

        return ret;



    /* parse IHDR (must be first chunk) */

    len = avio_rb32(pb);

    tag = avio_rl32(pb);

    if (len != 13 || tag != MKTAG('I', 'H', 'D', 'R'))

        return ret;



    st = avformat_new_stream(s, NULL);

    if (!st)

        return AVERROR(ENOMEM);



    /* set the timebase to something large enough (1/100,000 of second)

     * to hopefully cope with all sane frame durations */

    avpriv_set_pts_info(st, 64, 1, 100000);

    st->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;

    st->codecpar->codec_id   = AV_CODEC_ID_APNG;

    st->codecpar->width      = avio_rb32(pb);

    st->codecpar->height     = avio_rb32(pb);

    if ((ret = av_image_check_size(st->codecpar->width, st->codecpar->height, 0, s)) < 0)

        return ret;



    /* extradata will contain every chunk up to the first fcTL (excluded) */

    ctx->extra_data = av_malloc(len + 12 + AV_INPUT_BUFFER_PADDING_SIZE);

    if (!ctx->extra_data)

        return AVERROR(ENOMEM);

    ctx->extra_data_size = len + 12;

    AV_WB32(ctx->extra_data,    len);

    AV_WL32(ctx->extra_data+4,  tag);

    AV_WB32(ctx->extra_data+8,  st->codecpar->width);

    AV_WB32(ctx->extra_data+12, st->codecpar->height);

    if ((ret = avio_read(pb, ctx->extra_data+16, 9)) < 0)

        goto fail;



    while (!avio_feof(pb)) {

        if (acTL_found && ctx->num_play != 1) {

            int64_t size   = avio_size(pb);

            int64_t offset = avio_tell(pb);

            if (size < 0) {

                ret = size;

                goto fail;

            } else if (offset < 0) {

                ret = offset;

                goto fail;

            } else if ((ret = ffio_ensure_seekback(pb, size - offset)) < 0) {

                av_log(s, AV_LOG_WARNING, "Could not ensure seekback, will not loop\n");

                ctx->num_play = 1;

            }

        }

        if ((ctx->num_play == 1 || !acTL_found) &&

            ((ret = ffio_ensure_seekback(pb, 4 /* len */ + 4 /* tag */)) < 0))

            goto fail;



        len = avio_rb32(pb);

        if (len > 0x7fffffff) {

            ret = AVERROR_INVALIDDATA;

            goto fail;

        }



        tag = avio_rl32(pb);

        switch (tag) {

        case MKTAG('a', 'c', 'T', 'L'):

            if ((ret = avio_seek(pb, -8, SEEK_CUR)) < 0 ||

                (ret = append_extradata(ctx, pb, len + 12)) < 0)

                goto fail;

            acTL_found = 1;

            ctx->num_frames = AV_RB32(ctx->extra_data + ret + 8);

            ctx->num_play   = AV_RB32(ctx->extra_data + ret + 12);

            av_log(s, AV_LOG_DEBUG, "num_frames: %"PRIu32", num_play: %"PRIu32"\n",

                                    ctx->num_frames, ctx->num_play);

            break;

        case MKTAG('f', 'c', 'T', 'L'):

            if (!acTL_found) {

               ret = AVERROR_INVALIDDATA;

               goto fail;

            }

            if ((ret = avio_seek(pb, -8, SEEK_CUR)) < 0)

                goto fail;

            return 0;

        default:

            if ((ret = avio_seek(pb, -8, SEEK_CUR)) < 0 ||

                (ret = append_extradata(ctx, pb, len + 12)) < 0)

                goto fail;

        }

    }



fail:

    if (ctx->extra_data_size) {

        av_freep(&ctx->extra_data);

        ctx->extra_data_size = 0;

    }

    return ret;

}
