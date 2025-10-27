static int crystalhd_receive_frame(AVCodecContext *avctx, AVFrame *frame)

{

    BC_STATUS bc_ret;

    BC_DTS_STATUS decoder_status = { 0, };

    CopyRet rec_ret;

    CHDContext *priv   = avctx->priv_data;

    HANDLE dev         = priv->dev;

    int got_frame = 0;



    av_log(avctx, AV_LOG_VERBOSE, "CrystalHD: receive_frame\n");



    bc_ret = DtsGetDriverStatus(dev, &decoder_status);

    if (bc_ret != BC_STS_SUCCESS) {

        av_log(avctx, AV_LOG_ERROR, "CrystalHD: GetDriverStatus failed\n");

        return -1;

    }



    if (decoder_status.ReadyListCount == 0) {

        av_log(avctx, AV_LOG_INFO, "CrystalHD: Insufficient frames ready. Returning\n");

        return AVERROR(EAGAIN);

    }



    rec_ret = receive_frame(avctx, frame, &got_frame);

    if (rec_ret == RET_ERROR) {

        return -1;

    } else if (got_frame == 0) {

        return AVERROR(EAGAIN);

    } else {

        return 0;

    }

}
