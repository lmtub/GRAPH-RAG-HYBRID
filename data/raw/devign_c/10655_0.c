static int wav_write_header(AVFormatContext *s)

{

    WAVContext *wav = s->priv_data;

    AVIOContext *pb = s->pb;

    int64_t fmt, fact;



    ffio_wfourcc(pb, "RIFF");

    avio_wl32(pb, 0); /* file length */

    ffio_wfourcc(pb, "WAVE");



    /* format header */

    fmt = ff_start_tag(pb, "fmt ");

    if (ff_put_wav_header(pb, s->streams[0]->codec) < 0) {

        av_log(s, AV_LOG_ERROR, "%s codec not supported in WAVE format\n",

               s->streams[0]->codec->codec ? s->streams[0]->codec->codec->name : "NONE");

        av_free(wav);

        return -1;

    }

    ff_end_tag(pb, fmt);



    if (s->streams[0]->codec->codec_tag != 0x01 /* hence for all other than PCM */

        && s->pb->seekable) {

        fact = ff_start_tag(pb, "fact");

        avio_wl32(pb, 0);

        ff_end_tag(pb, fact);

    }



    av_set_pts_info(s->streams[0], 64, 1, s->streams[0]->codec->sample_rate);

    wav->maxpts = wav->last_duration = 0;

    wav->minpts = INT64_MAX;



    /* data header */

    wav->data = ff_start_tag(pb, "data");



    avio_flush(pb);



    return 0;

}
