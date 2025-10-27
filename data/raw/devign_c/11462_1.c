static int rv10_decode_init(AVCodecContext *avctx)

{

    MpegEncContext *s = avctx->priv_data;

    static int done=0;



    MPV_decode_defaults(s);

    

    s->avctx= avctx;

    s->out_format = FMT_H263;

    s->codec_id= avctx->codec_id;



    s->width = avctx->width;

    s->height = avctx->height;



    switch(avctx->sub_id){

    case 0x10000000:

        s->rv10_version= 0;

        s->h263_long_vectors=0;

        s->low_delay=1;

        break;

    case 0x10002000:

        s->rv10_version= 3;

        s->h263_long_vectors=1;

        s->low_delay=1;

        s->obmc=1;

        break;

    case 0x10003000:

        s->rv10_version= 3;

        s->h263_long_vectors=1;

        s->low_delay=1;

        break;

    case 0x10003001:

        s->rv10_version= 3;

        s->h263_long_vectors=0;

        s->low_delay=1;

        break;

    case 0x20001000:

    case 0x20100001:

    case 0x20101001:

    case 0x20103001:

        s->low_delay=1;

        break;

    case 0x20200002:

    case 0x20201002:

    case 0x30202002:

    case 0x30203002:

        s->low_delay=0;

        s->avctx->has_b_frames=1;

        break;

    default:

        av_log(s->avctx, AV_LOG_ERROR, "unknown header %X\n", avctx->sub_id);

    }

//av_log(avctx, AV_LOG_DEBUG, "ver:%X\n", avctx->sub_id);

    if (MPV_common_init(s) < 0)

        return -1;



    h263_decode_init_vlc(s);



    /* init rv vlc */

    if (!done) {

        init_vlc(&rv_dc_lum, DC_VLC_BITS, 256, 

                 rv_lum_bits, 1, 1,

                 rv_lum_code, 2, 2);

        init_vlc(&rv_dc_chrom, DC_VLC_BITS, 256, 

                 rv_chrom_bits, 1, 1,

                 rv_chrom_code, 2, 2);

        done = 1;

    }

    

    avctx->pix_fmt = PIX_FMT_YUV420P;



    return 0;

}
