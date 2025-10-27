static int siff_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    SIFFContext *c = s->priv_data;

    int size;



    if (c->has_video){

        if (c->cur_frame >= c->frames)

            return AVERROR_EOF;

        if (c->curstrm == -1){

            c->pktsize = avio_rl32(s->pb) - 4;

            c->flags = avio_rl16(s->pb);

            c->gmcsize = (c->flags & VB_HAS_GMC) ? 4 : 0;

            if (c->gmcsize)

                avio_read(s->pb, c->gmc, c->gmcsize);

            c->sndsize = (c->flags & VB_HAS_AUDIO) ? avio_rl32(s->pb): 0;

            c->curstrm = !!(c->flags & VB_HAS_AUDIO);

        }



        if (!c->curstrm){

            size = c->pktsize - c->sndsize - c->gmcsize - 2;

            size = ffio_limit(s->pb, size);

            if(size < 0 || c->pktsize < c->sndsize)

                return AVERROR_INVALIDDATA;

            if (av_new_packet(pkt, size + c->gmcsize + 2) < 0)

                return AVERROR(ENOMEM);

            AV_WL16(pkt->data, c->flags);

            if (c->gmcsize)

                memcpy(pkt->data + 2, c->gmc, c->gmcsize);

            avio_read(s->pb, pkt->data + 2 + c->gmcsize, size);

            pkt->stream_index = 0;

            c->curstrm = -1;

        }else{

            if ((size = av_get_packet(s->pb, pkt, c->sndsize - 4)) < 0)

                return AVERROR(EIO);

            pkt->stream_index = 1;

            pkt->duration     = size;

            c->curstrm = 0;

        }

        if(!c->cur_frame || c->curstrm)

            pkt->flags |= AV_PKT_FLAG_KEY;

        if (c->curstrm == -1)

            c->cur_frame++;

    }else{

        size = av_get_packet(s->pb, pkt, c->block_align);

        if(!size)

            return AVERROR_EOF;

        if(size < 0)

            return AVERROR(EIO);

        pkt->duration = size;

    }

    return pkt->size;

}
