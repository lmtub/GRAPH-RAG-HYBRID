static int interleave_new_audio_packet(AVFormatContext *s, AVPacket *pkt,

                                       int stream_index, int flush)

{

    AVStream *st = s->streams[stream_index];

    AudioInterleaveContext *aic = st->priv_data;



    int size = FFMIN(av_fifo_size(aic->fifo), *aic->samples * aic->sample_size);

    if (!size || (!flush && size == av_fifo_size(aic->fifo)))

        return 0;



    av_new_packet(pkt, size);

    av_fifo_generic_read(aic->fifo, pkt->data, size, NULL);



    pkt->dts = pkt->pts = aic->dts;

    pkt->duration = av_rescale_q(*aic->samples, st->time_base, aic->time_base);

    pkt->stream_index = stream_index;

    aic->dts += pkt->duration;



    aic->samples++;

    if (!*aic->samples)

        aic->samples = aic->samples_per_frame;



    return size;

}
