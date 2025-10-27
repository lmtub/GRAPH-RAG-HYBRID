static int tta_decode_frame(AVCodecContext *avctx,

        void *data, int *data_size,

        AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    TTAContext *s = avctx->priv_data;

    int i;



    init_get_bits(&s->gb, buf, buf_size*8);

    {

        int32_t predictors[s->channels];

        TTAFilter filters[s->channels];

        TTARice rices[s->channels];

        int cur_chan = 0, framelen = s->frame_length;

        int32_t *p;



        // FIXME: seeking

        s->total_frames--;

        if (!s->total_frames && s->last_frame_length)

            framelen = s->last_frame_length;



        // init per channel states

        for (i = 0; i < s->channels; i++) {

            predictors[i] = 0;

            ttafilter_init(&(filters[i]), ttafilter_configs[s->bps-1][0], ttafilter_configs[s->bps-1][1]);

            rice_init(&(rices[i]), 10, 10);

        }



        for (p = s->decode_buffer; p < s->decode_buffer + (framelen * s->channels); p++) {

            int32_t *predictor = &(predictors[cur_chan]);

            TTAFilter *filter = &(filters[cur_chan]);

            TTARice *rice = &(rices[cur_chan]);

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



            if (k)

                value = (unary << k) + get_bits(&s->gb, k);

            else

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



#if 0

            // extract 32bit float from last two int samples

            if (s->is_float && ((p - data) & 1)) {

                uint32_t neg = *p & 0x80000000;

                uint32_t hi = *(p - 1);

                uint32_t lo = abs(*p) - 1;



                hi += (hi || lo) ? 0x3f80 : 0;

                // SWAP16: swap all the 16 bits

                *(p - 1) = (hi << 16) | SWAP16(lo) | neg;

            }

#endif



            /*if ((get_bits_count(&s->gb)+7)/8 > buf_size)

            {

                av_log(NULL, AV_LOG_INFO, "overread!!\n");

                break;

            }*/



            // flip channels

            if (cur_chan < (s->channels-1))

                cur_chan++;

            else {

                // decorrelate in case of stereo integer

                if (!s->is_float && (s->channels > 1)) {

                    int32_t *r = p - 1;

                    for (*p += *r / 2; r > p - s->channels; r--)

                        *r = *(r + 1) - *r;

                }

                cur_chan = 0;

            }

        }



        skip_bits(&s->gb, 32); // frame crc



        // convert to output buffer

        switch(s->bps) {

            case 2: {

                uint16_t *samples = data;

                for (p = s->decode_buffer; p < s->decode_buffer + (framelen * s->channels); p++) {

//                    *samples++ = (unsigned char)*p;

//                    *samples++ = (unsigned char)(*p >> 8);

                    *samples++ = *p;

                }

                *data_size = (uint8_t *)samples - (uint8_t *)data;

                break;

            }

            default:

                av_log(s->avctx, AV_LOG_ERROR, "Error, only 16bit samples supported!\n");

        }

    }



//    return get_bits_count(&s->gb)+7)/8;

    return buf_size;

}
