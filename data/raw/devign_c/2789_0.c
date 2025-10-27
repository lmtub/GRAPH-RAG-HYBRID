static int decode_frame_mp3on4(AVCodecContext * avctx,

                        void *data, int *data_size,

                        AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    MP3On4DecodeContext *s = avctx->priv_data;

    MPADecodeContext *m;

    int fsize, len = buf_size, out_size = 0;

    uint32_t header;

    OUT_INT *out_samples = data;

    OUT_INT *outptr, *bp;

    int fr, j, n;



    if(*data_size < MPA_FRAME_SIZE * MPA_MAX_CHANNELS * s->frames * sizeof(OUT_INT))

        return -1;



    *data_size = 0;

    // Discard too short frames

    if (buf_size < HEADER_SIZE)

        return -1;



    // If only one decoder interleave is not needed

    outptr = s->frames == 1 ? out_samples : s->decoded_buf;



    avctx->bit_rate = 0;



    for (fr = 0; fr < s->frames; fr++) {

        fsize = AV_RB16(buf) >> 4;

        fsize = FFMIN3(fsize, len, MPA_MAX_CODED_FRAME_SIZE);

        m = s->mp3decctx[fr];

        assert (m != NULL);



        header = (AV_RB32(buf) & 0x000fffff) | s->syncword; // patch header



        if (ff_mpa_check_header(header) < 0) // Bad header, discard block

            break;



        avpriv_mpegaudio_decode_header((MPADecodeHeader *)m, header);

        out_size += mp_decode_frame(m, outptr, buf, fsize);

        buf += fsize;

        len -= fsize;



        if(s->frames > 1) {

            n = m->avctx->frame_size*m->nb_channels;

            /* interleave output data */

            bp = out_samples + s->coff[fr];

            if(m->nb_channels == 1) {

                for(j = 0; j < n; j++) {

                    *bp = s->decoded_buf[j];

                    bp += avctx->channels;

                }

            } else {

                for(j = 0; j < n; j++) {

                    bp[0] = s->decoded_buf[j++];

                    bp[1] = s->decoded_buf[j];

                    bp += avctx->channels;

                }

            }

        }

        avctx->bit_rate += m->bit_rate;

    }



    /* update codec info */

    avctx->sample_rate = s->mp3decctx[0]->sample_rate;



    *data_size = out_size;

    return buf_size;

}
