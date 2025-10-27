static int ws_snd_decode_frame(AVCodecContext *avctx,

                void *data, int *data_size,

                AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

//    WSSNDContext *c = avctx->priv_data;



    int in_size, out_size;

    int sample = 128;

    int i;

    uint8_t *samples = data;



    if (!buf_size)

        return 0;



    out_size = AV_RL16(&buf[0]);

    in_size = AV_RL16(&buf[2]);

    buf += 4;



    if (out_size > *data_size) {

        av_log(avctx, AV_LOG_ERROR, "Frame is too large to fit in buffer\n");

        return -1;

    }

    if (in_size > buf_size) {

        av_log(avctx, AV_LOG_ERROR, "Frame data is larger than input buffer\n");

        return -1;

    }



    *data_size = out_size;



    if (in_size == out_size) {

        for (i = 0; i < out_size; i++)

            *samples++ = *buf++;

        return buf_size;

    }



    while (out_size > 0) {

        int code;

        uint8_t count;

        code = (*buf) >> 6;

        count = (*buf) & 0x3F;

        buf++;

        switch(code) {

        case 0: /* ADPCM 2-bit */

            for (count++; count > 0; count--) {

                code = *buf++;

                sample += ws_adpcm_2bit[code & 0x3];

                sample = av_clip_uint8(sample);

                *samples++ = sample;

                sample += ws_adpcm_2bit[(code >> 2) & 0x3];

                sample = av_clip_uint8(sample);

                *samples++ = sample;

                sample += ws_adpcm_2bit[(code >> 4) & 0x3];

                sample = av_clip_uint8(sample);

                *samples++ = sample;

                sample += ws_adpcm_2bit[(code >> 6) & 0x3];

                sample = av_clip_uint8(sample);

                *samples++ = sample;

                out_size -= 4;

            }

            break;

        case 1: /* ADPCM 4-bit */

            for (count++; count > 0; count--) {

                code = *buf++;

                sample += ws_adpcm_4bit[code & 0xF];

                sample = av_clip_uint8(sample);

                *samples++ = sample;

                sample += ws_adpcm_4bit[code >> 4];

                sample = av_clip_uint8(sample);

                *samples++ = sample;

                out_size -= 2;

            }

            break;

        case 2: /* no compression */

            if (count & 0x20) { /* big delta */

                int8_t t;

                t = count;

                t <<= 3;

                sample += t >> 3;

                sample = av_clip_uint8(sample);

                *samples++ = sample;

                out_size--;

            } else { /* copy */

                for (count++; count > 0; count--) {

                    *samples++ = *buf++;

                    out_size--;

                }

                sample = buf[-1];

            }

            break;

        default: /* run */

            for(count++; count > 0; count--) {

                *samples++ = sample;

                out_size--;

            }

        }

    }



    return buf_size;

}
