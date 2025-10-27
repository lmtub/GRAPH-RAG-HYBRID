static av_cold int peak_init_writer(AVFormatContext *s)

{

    WAVMuxContext *wav = s->priv_data;

    AVCodecContext *enc = s->streams[0]->codec;



    if (enc->codec_id != AV_CODEC_ID_PCM_S8 &&

        enc->codec_id != AV_CODEC_ID_PCM_S16LE &&

        enc->codec_id != AV_CODEC_ID_PCM_U8 &&

        enc->codec_id != AV_CODEC_ID_PCM_U16LE) {

        av_log(s, AV_LOG_ERROR, "%s codec not supported for Peak Chunk\n",

               s->streams[0]->codec->codec ? s->streams[0]->codec->codec->name : "NONE");

        return -1;

    }



    wav->peak_bps = av_get_bits_per_sample(enc->codec_id) / 8;



    if (wav->peak_bps == 1 && wav->peak_format == PEAK_FORMAT_UINT16) {

        av_log(s, AV_LOG_ERROR,

               "Writing 16 bit peak for 8 bit audio does not make sense\n");

        return AVERROR(EINVAL);

    }



    wav->peak_maxpos = av_mallocz(enc->channels * sizeof(*wav->peak_maxpos));

    if (!wav->peak_maxpos)

        goto nomem;

    wav->peak_maxneg = av_mallocz(enc->channels * sizeof(*wav->peak_maxneg));

    if (!wav->peak_maxneg)

        goto nomem;



    wav->peak_output = av_malloc(PEAK_BUFFER_SIZE);

    if (!wav->peak_output)

        goto nomem;



    wav->peak_outbuf_size = PEAK_BUFFER_SIZE;



    return 0;



nomem:

    av_log(s, AV_LOG_ERROR, "Out of memory\n");

    peak_free_buffers(s);

    return AVERROR(ENOMEM);

}
