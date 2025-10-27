static void probe_codec(AVFormatContext *s, AVStream *st, const AVPacket *pkt)

{

    if(st->codec->codec_id == CODEC_ID_PROBE){

        AVProbeData *pd = &st->probe_data;

        av_log(s, AV_LOG_DEBUG, "probing stream %d\n", st->index);

        --st->probe_packets;



        pd->buf = av_realloc(pd->buf, pd->buf_size+pkt->size+AVPROBE_PADDING_SIZE);

        memcpy(pd->buf+pd->buf_size, pkt->data, pkt->size);

        pd->buf_size += pkt->size;

        memset(pd->buf+pd->buf_size, 0, AVPROBE_PADDING_SIZE);



        if(av_log2(pd->buf_size) != av_log2(pd->buf_size - pkt->size)){

            //FIXME we do not reduce score to 0 for the case of running out of buffer space in bytes

            set_codec_from_probe_data(s, st, pd, st->probe_packets > 0 ? AVPROBE_SCORE_MAX/4 : 0);

            if(st->codec->codec_id != CODEC_ID_PROBE){

                pd->buf_size=0;

                av_freep(&pd->buf);

                av_log(s, AV_LOG_DEBUG, "probed stream %d\n", st->index);

            }

        }

    }

}
