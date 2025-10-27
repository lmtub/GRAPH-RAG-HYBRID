static int dxa_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    DXAContext *c = s->priv_data;

    int ret;

    uint32_t size;

    uint8_t buf[DXA_EXTRA_SIZE], pal[768+4];

    int pal_size = 0;



    if(!c->readvid && c->has_sound && c->bytes_left){

        c->readvid = 1;

        avio_seek(s->pb, c->wavpos, SEEK_SET);

        size = FFMIN(c->bytes_left, c->bpc);

        ret = av_get_packet(s->pb, pkt, size);

        pkt->stream_index = 1;

        if(ret != size)

            return AVERROR(EIO);

        c->bytes_left -= size;

        c->wavpos = avio_tell(s->pb);

        return 0;

    }

    avio_seek(s->pb, c->vidpos, SEEK_SET);

    while(!url_feof(s->pb) && c->frames){

        avio_read(s->pb, buf, 4);

        switch(AV_RL32(buf)){

        case MKTAG('N', 'U', 'L', 'L'):

            if(av_new_packet(pkt, 4 + pal_size) < 0)

                return AVERROR(ENOMEM);

            pkt->stream_index = 0;

            if(pal_size) memcpy(pkt->data, pal, pal_size);

            memcpy(pkt->data + pal_size, buf, 4);

            c->frames--;

            c->vidpos = avio_tell(s->pb);

            c->readvid = 0;

            return 0;

        case MKTAG('C', 'M', 'A', 'P'):

            pal_size = 768+4;

            memcpy(pal, buf, 4);

            avio_read(s->pb, pal + 4, 768);

            break;

        case MKTAG('F', 'R', 'A', 'M'):

            avio_read(s->pb, buf + 4, DXA_EXTRA_SIZE - 4);

            size = AV_RB32(buf + 5);

            if(size > 0xFFFFFF){

                av_log(s, AV_LOG_ERROR, "Frame size is too big: %d\n", size);

                return AVERROR_INVALIDDATA;

            }

            if(av_new_packet(pkt, size + DXA_EXTRA_SIZE + pal_size) < 0)

                return AVERROR(ENOMEM);

            memcpy(pkt->data + pal_size, buf, DXA_EXTRA_SIZE);

            ret = avio_read(s->pb, pkt->data + DXA_EXTRA_SIZE + pal_size, size);

            if(ret != size){

                av_free_packet(pkt);

                return AVERROR(EIO);

            }

            if(pal_size) memcpy(pkt->data, pal, pal_size);

            pkt->stream_index = 0;

            c->frames--;

            c->vidpos = avio_tell(s->pb);

            c->readvid = 0;

            return 0;

        default:

            av_log(s, AV_LOG_ERROR, "Unknown tag %c%c%c%c\n", buf[0], buf[1], buf[2], buf[3]);

            return AVERROR_INVALIDDATA;

        }

    }

    return AVERROR_EOF;

}
