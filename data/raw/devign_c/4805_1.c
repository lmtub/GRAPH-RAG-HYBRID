static int pcm_decode_frame(AVCodecContext *avctx,

                            void *data, int *data_size,

                            uint8_t *buf, int buf_size)

{

    PCMDecode *s = avctx->priv_data;

    int c, n;

    short *samples;

    uint8_t *src, *src2[MAX_CHANNELS];



    samples = data;

    src = buf;



    n= av_get_bits_per_sample(avctx->codec_id)/8;

    if((n && buf_size % n) || avctx->channels > MAX_CHANNELS){

        av_log(avctx, AV_LOG_ERROR, "invalid PCM packet\n");

        return -1;

    }



    buf_size= FFMIN(buf_size, *data_size/2);

    *data_size=0;



    n = buf_size/avctx->channels;

    for(c=0;c<avctx->channels;c++)

        src2[c] = &src[c*n];



    switch(avctx->codec->id) {

    case CODEC_ID_PCM_S32LE:

        decode_to16(4, 1, 0, &src, &samples, buf_size);

        break;

    case CODEC_ID_PCM_S32BE:

        decode_to16(4, 0, 0, &src, &samples, buf_size);

        break;

    case CODEC_ID_PCM_U32LE:

        decode_to16(4, 1, 1, &src, &samples, buf_size);

        break;

    case CODEC_ID_PCM_U32BE:

        decode_to16(4, 0, 1, &src, &samples, buf_size);

        break;

    case CODEC_ID_PCM_S24LE:

        decode_to16(3, 1, 0, &src, &samples, buf_size);

        break;

    case CODEC_ID_PCM_S24BE:

        decode_to16(3, 0, 0, &src, &samples, buf_size);

        break;

    case CODEC_ID_PCM_U24LE:

        decode_to16(3, 1, 1, &src, &samples, buf_size);

        break;

    case CODEC_ID_PCM_U24BE:

        decode_to16(3, 0, 1, &src, &samples, buf_size);

        break;

    case CODEC_ID_PCM_S24DAUD:

        n = buf_size / 3;

        for(;n>0;n--) {

          uint32_t v = bytestream_get_be24(&src);

          v >>= 4; // sync flags are here

          *samples++ = ff_reverse[(v >> 8) & 0xff] +

                       (ff_reverse[v & 0xff] << 8);

        }

        break;

    case CODEC_ID_PCM_S16LE:

        n = buf_size >> 1;

        for(;n>0;n--) {

            *samples++ = bytestream_get_le16(&src);

        }

        break;

    case CODEC_ID_PCM_S16LE_PLANAR:

        for(n>>=1;n>0;n--)

            for(c=0;c<avctx->channels;c++)

                *samples++ = bytestream_get_le16(&src2[c]);

        src = src2[avctx->channels-1];

        break;

    case CODEC_ID_PCM_S16BE:

        n = buf_size >> 1;

        for(;n>0;n--) {

            *samples++ = bytestream_get_be16(&src);

        }

        break;

    case CODEC_ID_PCM_U16LE:

        n = buf_size >> 1;

        for(;n>0;n--) {

            *samples++ = bytestream_get_le16(&src) - 0x8000;

        }

        break;

    case CODEC_ID_PCM_U16BE:

        n = buf_size >> 1;

        for(;n>0;n--) {

            *samples++ = bytestream_get_be16(&src) - 0x8000;

        }

        break;

    case CODEC_ID_PCM_S8:

        n = buf_size;

        for(;n>0;n--) {

            *samples++ = *src++ << 8;

        }

        break;

    case CODEC_ID_PCM_U8:

        n = buf_size;

        for(;n>0;n--) {

            *samples++ = ((int)*src++ - 128) << 8;

        }

        break;

    case CODEC_ID_PCM_ZORK:

        n = buf_size;

        for(;n>0;n--) {

            int x= *src++;

            if(x&128) x-= 128;

            else      x = -x;

            *samples++ = x << 8;

        }

        break;

    case CODEC_ID_PCM_ALAW:

    case CODEC_ID_PCM_MULAW:

        n = buf_size;

        for(;n>0;n--) {

            *samples++ = s->table[*src++];

        }

        break;

    default:

        return -1;

    }

    *data_size = (uint8_t *)samples - (uint8_t *)data;

    return src - buf;

}
