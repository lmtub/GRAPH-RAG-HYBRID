static av_cold int xvid_encode_close(AVCodecContext *avctx)

{

    struct xvid_context *x = avctx->priv_data;



    if (x->encoder_handle) {

        xvid_encore(x->encoder_handle, XVID_ENC_DESTROY, NULL, NULL);

        x->encoder_handle = NULL;

    }



    av_frame_free(&avctx->coded_frame);

    av_freep(&avctx->extradata);

    if (x->twopassbuffer) {

        av_free(x->twopassbuffer);

        av_free(x->old_twopassbuffer);

    }

    av_free(x->twopassfile);

    av_free(x->intra_matrix);

    av_free(x->inter_matrix);



    return 0;

}
