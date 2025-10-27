static int adts_write_header(AVFormatContext *s)

{

    ADTSContext *adts = s->priv_data;

    AVCodecContext *avc = s->streams[0]->codec;



    if(avc->extradata_size > 0)

        decode_extradata(adts, avc->extradata, avc->extradata_size);



    return 0;

}
