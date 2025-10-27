AVStream *add_av_stream1(FFStream *stream, AVCodecContext *codec)

{

    AVStream *fst;



    fst = av_mallocz(sizeof(AVStream));

    if (!fst)

        return NULL;

    fst->priv_data = av_mallocz(sizeof(FeedData));

    memcpy(&fst->codec, codec, sizeof(AVCodecContext));


    stream->streams[stream->nb_streams++] = fst;

    return fst;

}