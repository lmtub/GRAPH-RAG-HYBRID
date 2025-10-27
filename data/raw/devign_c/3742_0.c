static int sipr_decode_frame(AVCodecContext *avctx, void *datap,

                             int *data_size, AVPacket *avpkt)

{

    SiprContext *ctx = avctx->priv_data;

    const uint8_t *buf=avpkt->data;

    SiprParameters parm;

    const SiprModeParam *mode_par = &modes[ctx->mode];

    GetBitContext gb;

    float *data = datap;

    int subframe_size = ctx->mode == MODE_16k ? L_SUBFR_16k : SUBFR_SIZE;

    int i;



    ctx->avctx = avctx;

    if (avpkt->size < (mode_par->bits_per_frame >> 3)) {

        av_log(avctx, AV_LOG_ERROR,

               "Error processing packet: packet size (%d) too small\n",

               avpkt->size);



        *data_size = 0;

        return -1;

    }

    if (*data_size < subframe_size * mode_par->subframe_count * sizeof(float)) {

        av_log(avctx, AV_LOG_ERROR,

               "Error processing packet: output buffer (%d) too small\n",

               *data_size);



        *data_size = 0;

        return -1;

    }



    init_get_bits(&gb, buf, mode_par->bits_per_frame);



    for (i = 0; i < mode_par->frames_per_packet; i++) {

        decode_parameters(&parm, &gb, mode_par);



        if (ctx->mode == MODE_16k)

            ff_sipr_decode_frame_16k(ctx, &parm, data);

        else

            decode_frame(ctx, &parm, data);



        data += subframe_size * mode_par->subframe_count;

    }



    *data_size = mode_par->frames_per_packet * subframe_size *

        mode_par->subframe_count * sizeof(float);



    return mode_par->bits_per_frame >> 3;

}
