static int tta_decode_frame(AVCodecContext *avctx, void *data,

                            int *got_frame_ptr, AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    TTAContext *s = avctx->priv_data;

    int i, ret;

    int cur_chan = 0, framelen = s->frame_length;

    int32_t *p;



    init_get_bits(&s->gb, buf, buf_size*8);



    // FIXME: seeking

    s->total_frames--;

    if (!s->total_frames && s->last_frame_length)

        framelen = s->last_frame_length;



    /* get output buffer */

    s->frame.nb_samples = framelen;

    if ((ret = avctx->get_buffer(avctx, &s->frame)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return ret;

    }



    // decode directly to output buffer for 24-bit sample format

    if (s->bps == 3)

        s->decode_buffer = (int32_t *)s->frame.data[0];



    // init per channel states

    for (i = 0; i < s->channels; i++) {

        s->ch_ctx[i].predictor = 0;

        ttafilter_init(&s->ch_ctx[i].filter, ttafilter_configs[s->bps-1][0], ttafilter_configs[s->bps-1][1]);

        rice_init(&s->ch_ctx[i].rice, 10, 10);

    }



    for (p = s->decode_buffer; p < s->decode_buffer + (framelen * s->channels); p++) {

        int32_t *predictor = &s->ch_ctx[cur_chan].predictor;

        TTAFilter *filter = &s->ch_ctx[cur_chan].filter;

        TTARice *rice = &s->ch_ctx[cur_chan].rice;

        uint32_t unary, depth, k;

        int32_t value;



        unary = tta_get_unary(&s->gb);



        if (unary == 0) {

            depth = 0;

            k = rice->k0;

        } else {

            depth = 1;

            k = rice->k1;

            unary--;

        }



        if (get_bits_left(&s->gb) < k)

            return -1;



        if (k) {

            if (k > MIN_CACHE_BITS)

                return -1;

            value = (unary << k) + get_bits(&s->gb, k);

        } else

            value = unary;



        // FIXME: copy paste from original

        switch (depth) {

        case 1:

            rice->sum1 += value - (rice->sum1 >> 4);

            if (rice->k1 > 0 && rice->sum1 < shift_16[rice->k1])

                rice->k1--;

            else if(rice->sum1 > shift_16[rice->k1 + 1])

                rice->k1++;

            value += shift_1[rice->k0];

        default:

            rice->sum0 += value - (rice->sum0 >> 4);

            if (rice->k0 > 0 && rice->sum0 < shift_16[rice->k0])

                rice->k0--;

            else if(rice->sum0 > shift_16[rice->k0 + 1])

                rice->k0++;

        }



        // extract coded value

#define UNFOLD(x) (((x)&1) ? (++(x)>>1) : (-(x)>>1))

        *p = UNFOLD(value);



        // run hybrid filter

        ttafilter_process(filter, p, 0);



        // fixed order prediction

#define PRED(x, k) (int32_t)((((uint64_t)x << k) - x) >> k)

        switch (s->bps) {

            case 1: *p += PRED(*predictor, 4); break;

            case 2:

            case 3: *p += PRED(*predictor, 5); break;

            case 4: *p += *predictor; break;

        }

        *predictor = *p;



        // flip channels

        if (cur_chan < (s->channels-1))

            cur_chan++;

        else {

            // decorrelate in case of stereo integer

            if (s->channels > 1) {

                int32_t *r = p - 1;

                for (*p += *r / 2; r > p - s->channels; r--)

                    *r = *(r + 1) - *r;

            }

            cur_chan = 0;

        }

    }



    if (get_bits_left(&s->gb) < 32)

        return -1;

    skip_bits_long(&s->gb, 32); // frame crc



    // convert to output buffer

    if (s->bps == 2) {

        int16_t *samples = (int16_t *)s->frame.data[0];

        for (p = s->decode_buffer; p < s->decode_buffer + (framelen * s->channels); p++)

            *samples++ = *p;

    } else {

        // shift samples for 24-bit sample format

        int32_t *samples = (int32_t *)s->frame.data[0];

        for (i = 0; i < framelen * s->channels; i++)

            *samples++ <<= 8;

        // reset decode buffer

        s->decode_buffer = NULL;

    }



    *got_frame_ptr   = 1;

    *(AVFrame *)data = s->frame;



    return buf_size;

}
