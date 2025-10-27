static int wv_read_block_header(AVFormatContext *ctx, AVIOContext *pb)

{

    WVContext *wc = ctx->priv_data;

    int ret;

    int rate, bpp, chan;

    uint32_t chmask, flags;



    wc->pos = avio_tell(pb);



    /* don't return bogus packets with the ape tag data */

    if (wc->apetag_start && wc->pos >= wc->apetag_start)

        return AVERROR_EOF;



    ret = avio_read(pb, wc->block_header, WV_HEADER_SIZE);

    if (ret != WV_HEADER_SIZE)

        return (ret < 0) ? ret : AVERROR_EOF;



    ret = ff_wv_parse_header(&wc->header, wc->block_header);

    if (ret < 0) {

        av_log(ctx, AV_LOG_ERROR, "Invalid block header.\n");

        return ret;

    }



    if (wc->header.version < 0x402 || wc->header.version > 0x410) {

        av_log(ctx, AV_LOG_ERROR, "Unsupported version %03X\n", wc->header.version);

        return AVERROR_PATCHWELCOME;

    }



    /* Blocks with zero samples don't contain actual audio information

     * and should be ignored */

    if (!wc->header.samples)

        return 0;

    // parse flags

    flags  = wc->header.flags;

    bpp    = ((flags & 3) + 1) << 3;

    chan   = 1 + !(flags & WV_MONO);

    chmask = flags & WV_MONO ? AV_CH_LAYOUT_MONO : AV_CH_LAYOUT_STEREO;

    rate   = wv_rates[(flags >> 23) & 0xF];

    wc->multichannel = !(wc->header.initial && wc->header.final);

    if (wc->multichannel) {

        chan   = wc->chan;

        chmask = wc->chmask;

    }

    if ((rate == -1 || !chan) && !wc->block_parsed) {

        int64_t block_end = avio_tell(pb) + wc->header.blocksize;

        if (!pb->seekable) {

            av_log(ctx, AV_LOG_ERROR,

                   "Cannot determine additional parameters\n");

            return AVERROR_INVALIDDATA;

        }

        while (avio_tell(pb) < block_end) {

            int id, size;

            id   = avio_r8(pb);

            size = (id & 0x80) ? avio_rl24(pb) : avio_r8(pb);

            size <<= 1;

            if (id & 0x40)

                size--;

            switch (id & 0x3F) {

            case 0xD:

                if (size <= 1) {

                    av_log(ctx, AV_LOG_ERROR,

                           "Insufficient channel information\n");

                    return AVERROR_INVALIDDATA;

                }

                chan = avio_r8(pb);

                switch (size - 2) {

                case 0:

                    chmask = avio_r8(pb);

                    break;

                case 1:

                    chmask = avio_rl16(pb);

                    break;

                case 2:

                    chmask = avio_rl24(pb);

                    break;

                case 3:

                    chmask = avio_rl32(pb);

                    break;

                case 5:

                    avio_skip(pb, 1);

                    chan  |= (avio_r8(pb) & 0xF) << 8;

                    chmask = avio_rl24(pb);

                    break;

                default:

                    av_log(ctx, AV_LOG_ERROR,

                           "Invalid channel info size %d\n", size);

                    return AVERROR_INVALIDDATA;

                }

                break;

            case 0x27:

                rate = avio_rl24(pb);

                break;

            default:

                avio_skip(pb, size);

            }

            if (id & 0x40)

                avio_skip(pb, 1);

        }

        if (rate == -1) {

            av_log(ctx, AV_LOG_ERROR,

                   "Cannot determine custom sampling rate\n");

            return AVERROR_INVALIDDATA;

        }

        avio_seek(pb, block_end - wc->header.blocksize, SEEK_SET);

    }

    if (!wc->bpp)

        wc->bpp    = bpp;

    if (!wc->chan)

        wc->chan   = chan;

    if (!wc->chmask)

        wc->chmask = chmask;

    if (!wc->rate)

        wc->rate   = rate;



    if (flags && bpp != wc->bpp) {

        av_log(ctx, AV_LOG_ERROR,

               "Bits per sample differ, this block: %i, header block: %i\n",

               bpp, wc->bpp);

        return AVERROR_INVALIDDATA;

    }

    if (flags && !wc->multichannel && chan != wc->chan) {

        av_log(ctx, AV_LOG_ERROR,

               "Channels differ, this block: %i, header block: %i\n",

               chan, wc->chan);

        return AVERROR_INVALIDDATA;

    }

    if (flags && rate != -1 && rate != wc->rate) {

        av_log(ctx, AV_LOG_ERROR,

               "Sampling rate differ, this block: %i, header block: %i\n",

               rate, wc->rate);

        return AVERROR_INVALIDDATA;

    }

    return 0;

}
