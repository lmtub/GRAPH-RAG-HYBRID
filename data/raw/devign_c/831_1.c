static int h263_decode_init(AVCodecContext *avctx)

{

    MpegEncContext *s = avctx->priv_data;

    int i;



    s->avctx = avctx;

    s->out_format = FMT_H263;



    s->width = avctx->width;

    s->height = avctx->height;



    /* select sub codec */

    switch(avctx->codec->id) {

    case CODEC_ID_H263:

        s->gob_number = 0;

        s->first_gob_line = 0;

        break;

    case CODEC_ID_MPEG4:

        s->time_increment_bits = 4; /* default value for broken headers */

        s->h263_pred = 1;

        s->has_b_frames = 1; //default, might be overriden in the vol header during header parsing

        break;

    case CODEC_ID_MSMPEG4V1:

        s->h263_msmpeg4 = 1;

        s->h263_pred = 1;

        s->msmpeg4_version=1;

        break;

    case CODEC_ID_MSMPEG4V2:

        s->h263_msmpeg4 = 1;

        s->h263_pred = 1;

        s->msmpeg4_version=2;

        break;

    case CODEC_ID_MSMPEG4V3:

        s->h263_msmpeg4 = 1;

        s->h263_pred = 1;

        s->msmpeg4_version=3;

        break;

    case CODEC_ID_WMV1:

        s->h263_msmpeg4 = 1;

        s->h263_pred = 1;

        s->msmpeg4_version=4;

        break;

    case CODEC_ID_H263I:

        s->h263_intel = 1;

        break;

    default:

        return -1;

    }



    /* for h263, we allocate the images after having read the header */

    if (avctx->codec->id != CODEC_ID_H263 && avctx->codec->id != CODEC_ID_MPEG4)

        if (MPV_common_init(s) < 0)

            return -1;



    /* XXX: suppress this matrix init, only needed because using mpeg1

       dequantize in mmx case */

    for(i=0;i<64;i++)

        s->non_intra_matrix[i] = default_non_intra_matrix[i];



    if (s->h263_msmpeg4)

        msmpeg4_decode_init_vlc(s);

    else

        h263_decode_init_vlc(s);

    

    return 0;

}
