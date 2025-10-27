static int dpcm_decode_frame(AVCodecContext *avctx,

                             void *data, int *data_size,

                             AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    DPCMContext *s = avctx->priv_data;

    int in, out = 0;

    int predictor[2];

    int ch = 0;

    int stereo = s->channels - 1;

    short *output_samples = data;

    int shift[2];

    unsigned char byte;

    short diff;



    if (!buf_size)

        return 0;



    // almost every DPCM variant expands one byte of data into two

    if(*data_size/2 < buf_size)

        return -1;



    switch(avctx->codec->id) {



    case CODEC_ID_ROQ_DPCM:

        if (stereo) {

            predictor[0] = buf[7] << 8;

            predictor[1] = buf[6] << 8;

        } else {

            predictor[0] = AV_RL16(&buf[6]);

        }

        SE_16BIT(predictor[0]);

        SE_16BIT(predictor[1]);



        /* decode the samples */

        for (in = 8, out = 0; in < buf_size; in++, out++) {

            predictor[ch] += s->roq_square_array[buf[in]];

            predictor[ch] = av_clip_int16(predictor[ch]);

            output_samples[out] = predictor[ch];



            /* toggle channel */

            ch ^= stereo;

        }

        break;



    case CODEC_ID_INTERPLAY_DPCM:

        in = 6;  /* skip over the stream mask and stream length */

        predictor[0] = AV_RL16(&buf[in]);

        in += 2;

        SE_16BIT(predictor[0])

        output_samples[out++] = predictor[0];

        if (stereo) {

            predictor[1] = AV_RL16(&buf[in]);

            in += 2;

            SE_16BIT(predictor[1])

            output_samples[out++] = predictor[1];

        }



        while (in < buf_size) {

            predictor[ch] += interplay_delta_table[buf[in++]];

            predictor[ch] = av_clip_int16(predictor[ch]);

            output_samples[out++] = predictor[ch];



            /* toggle channel */

            ch ^= stereo;

        }



        break;



    case CODEC_ID_XAN_DPCM:

        in = 0;

        shift[0] = shift[1] = 4;

        predictor[0] = AV_RL16(&buf[in]);

        in += 2;

        SE_16BIT(predictor[0]);

        if (stereo) {

            predictor[1] = AV_RL16(&buf[in]);

            in += 2;

            SE_16BIT(predictor[1]);

        }



        while (in < buf_size) {

            byte = buf[in++];

            diff = (byte & 0xFC) << 8;

            if ((byte & 0x03) == 3)

                shift[ch]++;

            else

                shift[ch] -= (2 * (byte & 3));

            /* saturate the shifter to a lower limit of 0 */

            if (shift[ch] < 0)

                shift[ch] = 0;



            diff >>= shift[ch];

            predictor[ch] += diff;



            predictor[ch] = av_clip_int16(predictor[ch]);

            output_samples[out++] = predictor[ch];



            /* toggle channel */

            ch ^= stereo;

        }

        break;

    case CODEC_ID_SOL_DPCM:

        in = 0;

        if (avctx->codec_tag != 3) {

            if(*data_size/4 < buf_size)

                return -1;

            while (in < buf_size) {

                int n1, n2;

                n1 = (buf[in] >> 4) & 0xF;

                n2 = buf[in++] & 0xF;

                s->sample[0] += s->sol_table[n1];

                if (s->sample[0] < 0)   s->sample[0] = 0;

                if (s->sample[0] > 255) s->sample[0] = 255;

                output_samples[out++] = (s->sample[0] - 128) << 8;

                s->sample[stereo] += s->sol_table[n2];

                if (s->sample[stereo] < 0)   s->sample[stereo] = 0;

                if (s->sample[stereo] > 255) s->sample[stereo] = 255;

                output_samples[out++] = (s->sample[stereo] - 128) << 8;

            }

        } else {

            while (in < buf_size) {

                int n;

                n = buf[in++];

                if (n & 0x80) s->sample[ch] -= s->sol_table[n & 0x7F];

                else s->sample[ch] += s->sol_table[n & 0x7F];

                s->sample[ch] = av_clip_int16(s->sample[ch]);

                output_samples[out++] = s->sample[ch];

                /* toggle channel */

                ch ^= stereo;

            }

        }

        break;

    }



    *data_size = out * sizeof(short);

    return buf_size;

}
