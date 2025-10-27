void ff_qsv_decode_reset(AVCodecContext *avctx, QSVContext *q)

{

    QSVFrame *cur;

    AVPacket pkt;

    int ret = 0;

    mfxVideoParam param = { { 0 } };



    if (q->reinit_pending) {

        close_decoder(q);

    } else if (q->engine_ready) {

        ret = MFXVideoDECODE_GetVideoParam(q->session, &param);

        if (ret < 0) {

            av_log(avctx, AV_LOG_ERROR, "MFX decode get param error %d\n", ret);

        }



        ret = MFXVideoDECODE_Reset(q->session, &param);

        if (ret < 0) {

            av_log(avctx, AV_LOG_ERROR, "MFX decode reset error %d\n", ret);

        }



        /* Free all frames*/

        cur = q->work_frames;

        while (cur) {

            q->work_frames = cur->next;

            av_frame_free(&cur->frame);

            av_freep(&cur);

            cur = q->work_frames;

        }

    }



    /* Reset output surfaces */

    av_fifo_reset(q->async_fifo);



    /* Reset input packets fifo */

    while (av_fifo_size(q->pkt_fifo)) {

        av_fifo_generic_read(q->pkt_fifo, &pkt, sizeof(pkt), NULL);

        av_packet_unref(&pkt);

    }



    /* Reset input bitstream fifo */

    av_fifo_reset(q->input_fifo);

}
