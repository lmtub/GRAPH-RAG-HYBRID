static int oggvorbis_encode_frame(AVCodecContext *avccontext,

				  unsigned char *packets,

			   int buf_size, void *data)

{

    OggVorbisContext *context = avccontext->priv_data ;

    float **buffer ;

    ogg_packet op ;

    signed char *audio = data ;

    int l, samples = OGGVORBIS_FRAME_SIZE ;



    buffer = vorbis_analysis_buffer(&context->vd, samples) ;



    if(context->vi.channels == 1) {

	for(l = 0 ; l < samples ; l++)

	    buffer[0][l]=((audio[l*2+1]<<8)|(0x00ff&(int)audio[l*2]))/32768.f;

    } else {

	for(l = 0 ; l < samples ; l++){

	    buffer[0][l]=((audio[l*4+1]<<8)|(0x00ff&(int)audio[l*4]))/32768.f;

	    buffer[1][l]=((audio[l*4+3]<<8)|(0x00ff&(int)audio[l*4+2]))/32768.f;

	}

    }

    

    vorbis_analysis_wrote(&context->vd, samples) ; 



    while(vorbis_analysis_blockout(&context->vd, &context->vb) == 1) {

	vorbis_analysis(&context->vb, NULL);

	vorbis_bitrate_addblock(&context->vb) ;



	while(vorbis_bitrate_flushpacket(&context->vd, &op)) {

            memcpy(context->buffer + context->buffer_index, &op, sizeof(ogg_packet));

            context->buffer_index += sizeof(ogg_packet);

            memcpy(context->buffer + context->buffer_index, op.packet, op.bytes);

            context->buffer_index += op.bytes;

//            av_log(avccontext, AV_LOG_DEBUG, "e%d / %d\n", context->buffer_index, op.bytes);

	}

    }



    if(context->buffer_index){

        ogg_packet *op2= (ogg_packet*)context->buffer;

        op2->packet = context->buffer + sizeof(ogg_packet);

        l=  op2->bytes;

        

        memcpy(packets, op2->packet, l);

        context->buffer_index -= l + sizeof(ogg_packet);

        memcpy(context->buffer, context->buffer + l + sizeof(ogg_packet), context->buffer_index);

        

//        av_log(avccontext, AV_LOG_DEBUG, "E%d\n", l);

        return l;

    }



    return 0;

}
