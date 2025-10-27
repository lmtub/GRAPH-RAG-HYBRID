static void probe_codec(AVFormatContext *s, AVStream *st, const AVPacket *pkt)

{

    if(st->request_probe>0){

        AVProbeData *pd = &st->probe_data;

        int end;

        av_log(s, AV_LOG_DEBUG, "probing stream %d pp:%d\n", st->index, st->probe_packets);

        --st->probe_packets;



        if (pkt) {

            pd->buf = av_realloc(pd->buf, pd->buf_size+pkt->size+AVPROBE_PADDING_SIZE);

            memcpy(pd->buf+pd->buf_size, pkt->data, pkt->size);

            pd->buf_size += pkt->size;

            memset(pd->buf+pd->buf_size, 0, AVPROBE_PADDING_SIZE);

        } else {

            st->probe_packets = 0;

        }



        end=    s->raw_packet_buffer_remaining_size <= 0

                || st->probe_packets<=0;



        if(end || av_log2(pd->buf_size) != av_log2(pd->buf_size - pkt->size)){

            int score= set_codec_from_probe_data(s, st, pd);

            if(    (st->codec->codec_id != CODEC_ID_NONE && score > AVPROBE_SCORE_MAX/4)

                || end){

                pd->buf_size=0;

                av_freep(&pd->buf);

                st->request_probe= -1;

                if(st->codec->codec_id != CODEC_ID_NONE){

                    av_log(s, AV_LOG_DEBUG, "probed stream %d\n", st->index);

                }else

                    av_log(s, AV_LOG_WARNING, "probed stream %d failed\n", st->index);

            }

        }

    }

}
