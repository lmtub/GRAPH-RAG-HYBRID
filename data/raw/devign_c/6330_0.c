static AVStream *add_av_stream1(FFStream *stream, AVCodecContext *codec, int copy)

{

    AVStream *fst;



    fst = av_mallocz(sizeof(AVStream));

    if (!fst)

        return NULL;

    if (copy) {

        fst->codec= avcodec_alloc_context();

        memcpy(fst->codec, codec, sizeof(AVCodecContext));

        if (codec->extradata_size) {

            fst->codec->extradata = av_malloc(codec->extradata_size);

            memcpy(fst->codec->extradata, codec->extradata,

                codec->extradata_size);

        }

    } else {

        /* live streams must use the actual feed's codec since it may be

         * updated later to carry extradata needed by the streams.

         */

        fst->codec = codec;

    }

    fst->priv_data = av_mallocz(sizeof(FeedData));

    fst->index = stream->nb_streams;

    av_set_pts_info(fst, 33, 1, 90000);

    fst->sample_aspect_ratio = (AVRational){0,1};

    stream->streams[stream->nb_streams++] = fst;

    return fst;

}
