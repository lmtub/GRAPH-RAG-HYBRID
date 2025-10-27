static int adpcm_encode_frame(AVCodecContext *avctx,

                            unsigned char *frame, int buf_size, void *data)

{

    int n, i, st;

    short *samples;

    unsigned char *dst;

    ADPCMContext *c = avctx->priv_data;



    dst = frame;

    samples = (short *)data;

    st= avctx->channels == 2;

/*    n = (BLKSIZE - 4 * avctx->channels) / (2 * 8 * avctx->channels); */



    switch(avctx->codec->id) {

    case CODEC_ID_ADPCM_IMA_QT: /* XXX: can't test until we get .mov writer */

        break;

    case CODEC_ID_ADPCM_IMA_WAV:

        n = avctx->frame_size / 8;

            c->status[0].prev_sample = (signed short)samples[0]; /* XXX */

/*            c->status[0].step_index = 0; *//* XXX: not sure how to init the state machine */

            bytestream_put_le16(&dst, c->status[0].prev_sample);

            *dst++ = (unsigned char)c->status[0].step_index;

            *dst++ = 0; /* unknown */

            samples++;

            if (avctx->channels == 2) {

                c->status[1].prev_sample = (signed short)samples[1];

/*                c->status[1].step_index = 0; */

                bytestream_put_le16(&dst, c->status[1].prev_sample);

                *dst++ = (unsigned char)c->status[1].step_index;

                *dst++ = 0;

                samples++;

            }



            /* stereo: 4 bytes (8 samples) for left, 4 bytes for right, 4 bytes left, ... */

            if(avctx->trellis > 0) {

                uint8_t buf[2][n*8];

                adpcm_compress_trellis(avctx, samples, buf[0], &c->status[0], n*8);

                if(avctx->channels == 2)

                    adpcm_compress_trellis(avctx, samples+1, buf[1], &c->status[1], n*8);

                for(i=0; i<n; i++) {

                    *dst++ = buf[0][8*i+0] | (buf[0][8*i+1] << 4);

                    *dst++ = buf[0][8*i+2] | (buf[0][8*i+3] << 4);

                    *dst++ = buf[0][8*i+4] | (buf[0][8*i+5] << 4);

                    *dst++ = buf[0][8*i+6] | (buf[0][8*i+7] << 4);

                    if (avctx->channels == 2) {

                        *dst++ = buf[1][8*i+0] | (buf[1][8*i+1] << 4);

                        *dst++ = buf[1][8*i+2] | (buf[1][8*i+3] << 4);

                        *dst++ = buf[1][8*i+4] | (buf[1][8*i+5] << 4);

                        *dst++ = buf[1][8*i+6] | (buf[1][8*i+7] << 4);

                    }

                }

            } else

            for (; n>0; n--) {

                *dst = adpcm_ima_compress_sample(&c->status[0], samples[0]) & 0x0F;

                *dst |= (adpcm_ima_compress_sample(&c->status[0], samples[avctx->channels]) << 4) & 0xF0;

                dst++;

                *dst = adpcm_ima_compress_sample(&c->status[0], samples[avctx->channels * 2]) & 0x0F;

                *dst |= (adpcm_ima_compress_sample(&c->status[0], samples[avctx->channels * 3]) << 4) & 0xF0;

                dst++;

                *dst = adpcm_ima_compress_sample(&c->status[0], samples[avctx->channels * 4]) & 0x0F;

                *dst |= (adpcm_ima_compress_sample(&c->status[0], samples[avctx->channels * 5]) << 4) & 0xF0;

                dst++;

                *dst = adpcm_ima_compress_sample(&c->status[0], samples[avctx->channels * 6]) & 0x0F;

                *dst |= (adpcm_ima_compress_sample(&c->status[0], samples[avctx->channels * 7]) << 4) & 0xF0;

                dst++;

                /* right channel */

                if (avctx->channels == 2) {

                    *dst = adpcm_ima_compress_sample(&c->status[1], samples[1]);

                    *dst |= adpcm_ima_compress_sample(&c->status[1], samples[3]) << 4;

                    dst++;

                    *dst = adpcm_ima_compress_sample(&c->status[1], samples[5]);

                    *dst |= adpcm_ima_compress_sample(&c->status[1], samples[7]) << 4;

                    dst++;

                    *dst = adpcm_ima_compress_sample(&c->status[1], samples[9]);

                    *dst |= adpcm_ima_compress_sample(&c->status[1], samples[11]) << 4;

                    dst++;

                    *dst = adpcm_ima_compress_sample(&c->status[1], samples[13]);

                    *dst |= adpcm_ima_compress_sample(&c->status[1], samples[15]) << 4;

                    dst++;

                }

                samples += 8 * avctx->channels;

            }

        break;

    case CODEC_ID_ADPCM_SWF:

    {

        int i;

        PutBitContext pb;

        init_put_bits(&pb, dst, buf_size*8);



        n = avctx->frame_size-1;



        //Store AdpcmCodeSize

        put_bits(&pb, 2, 2);                //Set 4bits flash adpcm format



        //Init the encoder state

        for(i=0; i<avctx->channels; i++){

            c->status[i].step_index = av_clip(c->status[i].step_index, 0, 63); // clip step so it fits 6 bits

            put_bits(&pb, 16, samples[i] & 0xFFFF);

            put_bits(&pb, 6, c->status[i].step_index);

            c->status[i].prev_sample = (signed short)samples[i];

        }



        if(avctx->trellis > 0) {

            uint8_t buf[2][n];

            adpcm_compress_trellis(avctx, samples+2, buf[0], &c->status[0], n);

            if (avctx->channels == 2)

                adpcm_compress_trellis(avctx, samples+3, buf[1], &c->status[1], n);

            for(i=0; i<n; i++) {

                put_bits(&pb, 4, buf[0][i]);

                if (avctx->channels == 2)

                    put_bits(&pb, 4, buf[1][i]);

            }

        } else {

            for (i=1; i<avctx->frame_size; i++) {

                put_bits(&pb, 4, adpcm_ima_compress_sample(&c->status[0], samples[avctx->channels*i]) & 0xF);

                if (avctx->channels == 2)

                    put_bits(&pb, 4, adpcm_ima_compress_sample(&c->status[1], samples[2*i+1]) & 0xF);

            }

        }

        flush_put_bits(&pb);

        dst += put_bits_count(&pb)>>3;

        break;

    }

    case CODEC_ID_ADPCM_MS:

        for(i=0; i<avctx->channels; i++){

            int predictor=0;



            *dst++ = predictor;

            c->status[i].coeff1 = AdaptCoeff1[predictor];

            c->status[i].coeff2 = AdaptCoeff2[predictor];

        }

        for(i=0; i<avctx->channels; i++){

            if (c->status[i].idelta < 16)

                c->status[i].idelta = 16;



            bytestream_put_le16(&dst, c->status[i].idelta);

        }

        for(i=0; i<avctx->channels; i++){

            c->status[i].sample1= *samples++;



            bytestream_put_le16(&dst, c->status[i].sample1);

        }

        for(i=0; i<avctx->channels; i++){

            c->status[i].sample2= *samples++;



            bytestream_put_le16(&dst, c->status[i].sample2);

        }



        if(avctx->trellis > 0) {

            int n = avctx->block_align - 7*avctx->channels;

            uint8_t buf[2][n];

            if(avctx->channels == 1) {

                n *= 2;

                adpcm_compress_trellis(avctx, samples, buf[0], &c->status[0], n);

                for(i=0; i<n; i+=2)

                    *dst++ = (buf[0][i] << 4) | buf[0][i+1];

            } else {

                adpcm_compress_trellis(avctx, samples, buf[0], &c->status[0], n);

                adpcm_compress_trellis(avctx, samples+1, buf[1], &c->status[1], n);

                for(i=0; i<n; i++)

                    *dst++ = (buf[0][i] << 4) | buf[1][i];

            }

        } else

        for(i=7*avctx->channels; i<avctx->block_align; i++) {

            int nibble;

            nibble = adpcm_ms_compress_sample(&c->status[ 0], *samples++)<<4;

            nibble|= adpcm_ms_compress_sample(&c->status[st], *samples++);

            *dst++ = nibble;

        }

        break;

    case CODEC_ID_ADPCM_YAMAHA:

        n = avctx->frame_size / 2;

        if(avctx->trellis > 0) {

            uint8_t buf[2][n*2];

            n *= 2;

            if(avctx->channels == 1) {

                adpcm_compress_trellis(avctx, samples, buf[0], &c->status[0], n);

                for(i=0; i<n; i+=2)

                    *dst++ = buf[0][i] | (buf[0][i+1] << 4);

            } else {

                adpcm_compress_trellis(avctx, samples, buf[0], &c->status[0], n);

                adpcm_compress_trellis(avctx, samples+1, buf[1], &c->status[1], n);

                for(i=0; i<n; i++)

                    *dst++ = buf[0][i] | (buf[1][i] << 4);

            }

        } else

        for (; n>0; n--) {

            for(i = 0; i < avctx->channels; i++) {

                int nibble;

                nibble  = adpcm_yamaha_compress_sample(&c->status[i], samples[i]);

                nibble |= adpcm_yamaha_compress_sample(&c->status[i], samples[i+avctx->channels]) << 4;

                *dst++ = nibble;

            }

            samples += 2 * avctx->channels;

        }

        break;

    default:

        return -1;

    }

    return dst - frame;

}
