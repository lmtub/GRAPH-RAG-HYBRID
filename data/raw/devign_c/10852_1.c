void register_avcodec(AVCodec *codec)

{

    AVCodec **p;


    p = &first_avcodec;

    while (*p != NULL) p = &(*p)->next;

    *p = codec;

    codec->next = NULL;

}