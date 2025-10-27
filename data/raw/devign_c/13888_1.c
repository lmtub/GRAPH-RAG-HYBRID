static int vorbis_parse(AVCodecParserContext *s1, AVCodecContext *avctx,

                        const uint8_t **poutbuf, int *poutbuf_size,

                        const uint8_t *buf, int buf_size)

{

    VorbisParseContext *s = s1->priv_data;

    int duration;



    if (!s->vp && avctx->extradata && avctx->extradata_size) {

        s->vp = av_vorbis_parse_init(avctx->extradata, avctx->extradata_size);

        if (!s->vp)

            goto end;

    }



    if ((duration = av_vorbis_parse_frame(s->vp, buf, buf_size)) >= 0)

        s1->duration = duration;



end:

    /* always return the full packet. this parser isn't doing any splitting or

       combining, only packet analysis */

    *poutbuf      = buf;

    *poutbuf_size = buf_size;

    return buf_size;

}
