static int vmdaudio_decode_frame(AVCodecContext *avctx,

                                 void *data, int *data_size,

                                 uint8_t *buf, int buf_size)

{

    VmdAudioContext *s = (VmdAudioContext *)avctx->priv_data;

    unsigned int sound_flags;

    unsigned char *output_samples = (unsigned char *)data;



    /* point to the start of the encoded data */

    unsigned char *p = buf + 16;

    unsigned char *p_end = buf + buf_size;



    if (buf_size < 16)

        return buf_size;



    if (buf[6] == 1) {

        /* the chunk contains audio */

        *data_size = vmdaudio_loadsound(s, output_samples, p, 0);

    } else if (buf[6] == 2) {

        /* the chunk contains audio and silence mixed together */

        sound_flags = LE_32(p);

        p += 4;



        /* do something with extrabufs here? */



        while (p < p_end) {

            if (sound_flags & 0x01)

                /* silence */

                *data_size += vmdaudio_loadsound(s, output_samples, p, 1);

            else {

                /* audio */

                *data_size += vmdaudio_loadsound(s, output_samples, p, 0);

                p += s->block_align;

            }

            output_samples += (s->block_align * s->bits / 8);

            sound_flags >>= 1;

        }

    } else if (buf[6] == 3) {

        /* silent chunk */

        *data_size = vmdaudio_loadsound(s, output_samples, p, 1);

    }



    return buf_size;

}
