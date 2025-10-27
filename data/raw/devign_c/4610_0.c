enum CodecID av_codec_get_id(const AVCodecTag *tags[4], unsigned int tag)

{

    int i;

    for(i=0; i<4 && tags[i]; i++){

        enum CodecID id= codec_get_id(tags[i], tag);

        if(id!=CODEC_ID_NONE) return id;

    }

    return CODEC_ID_NONE;

}
