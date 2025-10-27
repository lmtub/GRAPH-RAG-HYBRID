void avcodec_default_release_buffer(AVCodecContext *s, AVFrame *pic){

    int i;

    InternalBuffer *buf, *last;

    AVCodecInternal *avci = s->internal;



    assert(s->codec_type == AVMEDIA_TYPE_VIDEO);



    assert(pic->type==FF_BUFFER_TYPE_INTERNAL);

    assert(avci->buffer_count);



    if (avci->buffer) {

        buf = NULL; /* avoids warning */

        for (i = 0; i < avci->buffer_count; i++) { //just 3-5 checks so is not worth to optimize

            buf = &avci->buffer[i];

            if (buf->data[0] == pic->data[0])

                break;

        }

        assert(i < avci->buffer_count);

        avci->buffer_count--;

        last = &avci->buffer[avci->buffer_count];



        FFSWAP(InternalBuffer, *buf, *last);

    }



    for (i = 0; i < AV_NUM_DATA_POINTERS; i++) {

        pic->data[i]=NULL;

//        pic->base[i]=NULL;

    }

//printf("R%X\n", pic->opaque);



    if(s->debug&FF_DEBUG_BUFFERS)

        av_log(s, AV_LOG_DEBUG, "default_release_buffer called on pic %p, %d "

               "buffers used\n", pic, avci->buffer_count);

}
