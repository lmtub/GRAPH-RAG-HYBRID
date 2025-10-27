unsigned int av_codec_get_tag(const AVCodecTag *tags[4], enum CodecID id)

{

    int i;

    for(i=0; i<4 && tags[i]; i++){

        int tag= codec_get_tag(tags[i], id);

        if(tag) return tag;

    }

    return 0;

}
