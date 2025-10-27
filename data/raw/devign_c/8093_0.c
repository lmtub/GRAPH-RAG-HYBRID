static av_cold int oggvorbis_encode_init(AVCodecContext *avccontext)

{

    OggVorbisContext *context = avccontext->priv_data;

    ogg_packet header, header_comm, header_code;

    uint8_t *p;

    unsigned int offset;



    vorbis_info_init(&context->vi);

    if (oggvorbis_init_encoder(&context->vi, avccontext) < 0) {

        av_log(avccontext, AV_LOG_ERROR, "oggvorbis_encode_init: init_encoder failed\n");

        return -1;

    }

    vorbis_analysis_init(&context->vd, &context->vi);

    vorbis_block_init(&context->vd, &context->vb);



    vorbis_comment_init(&context->vc);

    vorbis_comment_add_tag(&context->vc, "encoder", LIBAVCODEC_IDENT);



    vorbis_analysis_headerout(&context->vd, &context->vc, &header,

                              &header_comm, &header_code);



    avccontext->extradata_size =

        1 + xiph_len(header.bytes) + xiph_len(header_comm.bytes) +

        header_code.bytes;

    p = avccontext->extradata =

            av_malloc(avccontext->extradata_size + FF_INPUT_BUFFER_PADDING_SIZE);

    p[0]    = 2;

    offset  = 1;

    offset += av_xiphlacing(&p[offset], header.bytes);

    offset += av_xiphlacing(&p[offset], header_comm.bytes);

    memcpy(&p[offset], header.packet, header.bytes);

    offset += header.bytes;

    memcpy(&p[offset], header_comm.packet, header_comm.bytes);

    offset += header_comm.bytes;

    memcpy(&p[offset], header_code.packet, header_code.bytes);

    offset += header_code.bytes;

    assert(offset == avccontext->extradata_size);



#if 0

    vorbis_block_clear(&context->vb);

    vorbis_dsp_clear(&context->vd);

    vorbis_info_clear(&context->vi);

#endif

    vorbis_comment_clear(&context->vc);



    avccontext->frame_size = OGGVORBIS_FRAME_SIZE;



    avccontext->coded_frame = avcodec_alloc_frame();



    return 0;

}
