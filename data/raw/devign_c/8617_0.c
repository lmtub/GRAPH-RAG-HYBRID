static int wc3_read_header(AVFormatContext *s,

                           AVFormatParameters *ap)

{

    Wc3DemuxContext *wc3 = s->priv_data;

    ByteIOContext *pb = s->pb;

    unsigned int fourcc_tag;

    unsigned int size;

    AVStream *st;

    unsigned char preamble[WC3_PREAMBLE_SIZE];

    int ret = 0;

    int current_palette = 0;

    int bytes_to_read;

    int i;

    unsigned char rotate;



    /* default context members */

    wc3->width = WC3_DEFAULT_WIDTH;

    wc3->height = WC3_DEFAULT_HEIGHT;

    wc3->palettes = NULL;

    wc3->palette_count = 0;

    wc3->pts = 0;

    wc3->video_stream_index = wc3->audio_stream_index = 0;



    /* skip the first 3 32-bit numbers */

    url_fseek(pb, 12, SEEK_CUR);



    /* traverse through the chunks and load the header information before

     * the first BRCH tag */

    if ((ret = get_buffer(pb, preamble, WC3_PREAMBLE_SIZE)) !=

        WC3_PREAMBLE_SIZE)

        return AVERROR(EIO);

    fourcc_tag = AV_RL32(&preamble[0]);

    size = (AV_RB32(&preamble[4]) + 1) & (~1);



    do {

        switch (fourcc_tag) {



        case SOND_TAG:

        case INDX_TAG:

            /* SOND unknown, INDX unnecessary; ignore both */

            url_fseek(pb, size, SEEK_CUR);

            break;



        case _PC__TAG:

            /* need the number of palettes */

            url_fseek(pb, 8, SEEK_CUR);

            if ((ret = get_buffer(pb, preamble, 4)) != 4)

                return AVERROR(EIO);

            wc3->palette_count = AV_RL32(&preamble[0]);

            if((unsigned)wc3->palette_count >= UINT_MAX / PALETTE_SIZE){

                wc3->palette_count= 0;

                return -1;

            }

            wc3->palettes = av_malloc(wc3->palette_count * PALETTE_SIZE);

            break;



        case BNAM_TAG:

            /* load up the name */

            if ((unsigned)size < 512)

                bytes_to_read = size;

            else

                bytes_to_read = 512;

            if ((ret = get_buffer(pb, s->title, bytes_to_read)) != bytes_to_read)

                return AVERROR(EIO);

            break;



        case SIZE_TAG:

            /* video resolution override */

            if ((ret = get_buffer(pb, preamble, WC3_PREAMBLE_SIZE)) !=

                WC3_PREAMBLE_SIZE)

                return AVERROR(EIO);

            wc3->width = AV_RL32(&preamble[0]);

            wc3->height = AV_RL32(&preamble[4]);

            break;



        case PALT_TAG:

            /* one of several palettes */

            if ((unsigned)current_palette >= wc3->palette_count)

                return AVERROR_INVALIDDATA;

            if ((ret = get_buffer(pb,

                &wc3->palettes[current_palette * PALETTE_SIZE],

                PALETTE_SIZE)) != PALETTE_SIZE)

                return AVERROR(EIO);



            /* transform the current palette in place */

            for (i = current_palette * PALETTE_SIZE;

                 i < (current_palette + 1) * PALETTE_SIZE; i++) {

                /* rotate each palette component left by 2 and use the result

                 * as an index into the color component table */

                rotate = ((wc3->palettes[i] << 2) & 0xFF) |

                         ((wc3->palettes[i] >> 6) & 0xFF);

                wc3->palettes[i] = wc3_pal_lookup[rotate];

            }

            current_palette++;

            break;



        default:

            av_log(s, AV_LOG_ERROR, "  unrecognized WC3 chunk: %c%c%c%c (0x%02X%02X%02X%02X)\n",

                preamble[0], preamble[1], preamble[2], preamble[3],

                preamble[0], preamble[1], preamble[2], preamble[3]);

            return AVERROR_INVALIDDATA;

            break;

        }



        if ((ret = get_buffer(pb, preamble, WC3_PREAMBLE_SIZE)) !=

            WC3_PREAMBLE_SIZE)

            return AVERROR(EIO);

        fourcc_tag = AV_RL32(&preamble[0]);

        /* chunk sizes are 16-bit aligned */

        size = (AV_RB32(&preamble[4]) + 1) & (~1);



    } while (fourcc_tag != BRCH_TAG);



    /* initialize the decoder streams */

    st = av_new_stream(s, 0);

    if (!st)

        return AVERROR(ENOMEM);

    av_set_pts_info(st, 33, 1, 90000);

    wc3->video_stream_index = st->index;

    st->codec->codec_type = CODEC_TYPE_VIDEO;

    st->codec->codec_id = CODEC_ID_XAN_WC3;

    st->codec->codec_tag = 0;  /* no fourcc */

    st->codec->width = wc3->width;

    st->codec->height = wc3->height;



    /* palette considerations */

    st->codec->palctrl = &wc3->palette_control;



    st = av_new_stream(s, 0);

    if (!st)

        return AVERROR(ENOMEM);

    av_set_pts_info(st, 33, 1, 90000);

    wc3->audio_stream_index = st->index;

    st->codec->codec_type = CODEC_TYPE_AUDIO;

    st->codec->codec_id = CODEC_ID_PCM_S16LE;

    st->codec->codec_tag = 1;

    st->codec->channels = WC3_AUDIO_CHANNELS;

    st->codec->bits_per_sample = WC3_AUDIO_BITS;

    st->codec->sample_rate = WC3_SAMPLE_RATE;

    st->codec->bit_rate = st->codec->channels * st->codec->sample_rate *

        st->codec->bits_per_sample;

    st->codec->block_align = WC3_AUDIO_BITS * WC3_AUDIO_CHANNELS;



    return 0;

}
