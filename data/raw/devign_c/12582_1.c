static av_cold int vorbis_encode_init(AVCodecContext *avccontext)

{

    vorbis_enc_context *venc = avccontext->priv_data;



    if (avccontext->channels != 2) {

        av_log(avccontext, AV_LOG_ERROR, "Current Libav Vorbis encoder only supports 2 channels.\n");

        return -1;

    }



    create_vorbis_context(venc, avccontext);



    if (avccontext->flags & CODEC_FLAG_QSCALE)

        venc->quality = avccontext->global_quality / (float)FF_QP2LAMBDA / 10.;

    else

        venc->quality = 0.03;

    venc->quality *= venc->quality;



    avccontext->extradata_size = put_main_header(venc, (uint8_t**)&avccontext->extradata);



    avccontext->frame_size     = 1 << (venc->log2_blocksize[0] - 1);



    avccontext->coded_frame            = avcodec_alloc_frame();

    avccontext->coded_frame->key_frame = 1;



    return 0;

}
