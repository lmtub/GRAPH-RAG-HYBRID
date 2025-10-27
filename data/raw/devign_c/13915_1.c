static int libschroedinger_encode_close(AVCodecContext *avctx)

{

    SchroEncoderParams *p_schro_params = avctx->priv_data;



    /* Close the encoder. */

    schro_encoder_free(p_schro_params->encoder);



    /* Free data in the output frame queue. */

    ff_schro_queue_free(&p_schro_params->enc_frame_queue,

                        libschroedinger_free_frame);





    /* Free the encoder buffer. */

    if (p_schro_params->enc_buf_size)

        av_freep(&p_schro_params->enc_buf);



    /* Free the video format structure. */

    av_freep(&p_schro_params->format);



    return 0;

}
