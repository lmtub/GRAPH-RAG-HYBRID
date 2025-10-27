static int mov_write_packet(AVFormatContext *s, AVPacket *pkt)

{

    MOVContext *mov = s->priv_data;

    ByteIOContext *pb = s->pb;

    MOVTrack *trk = &mov->tracks[pkt->stream_index];

    AVCodecContext *enc = trk->enc;

    unsigned int samplesInChunk = 0;

    int size= pkt->size;



    if (url_is_streamed(s->pb)) return 0; /* Can't handle that */

    if (!size) return 0; /* Discard 0 sized packets */



    if (enc->codec_id == CODEC_ID_AMR_NB) {

        /* We must find out how many AMR blocks there are in one packet */

        static uint16_t packed_size[16] =

            {13, 14, 16, 18, 20, 21, 27, 32, 6, 0, 0, 0, 0, 0, 0, 0};

        int len = 0;



        while (len < size && samplesInChunk < 100) {

            len += packed_size[(pkt->data[len] >> 3) & 0x0F];

            samplesInChunk++;

        }

        if(samplesInChunk > 1){

            av_log(s, AV_LOG_ERROR, "fatal error, input is not a single packet, implement a AVParser for it\n");

            return -1;

        }

    } else if (trk->sampleSize)

        samplesInChunk = size/trk->sampleSize;

    else

        samplesInChunk = 1;



    /* copy extradata if it exists */

    if (trk->vosLen == 0 && enc->extradata_size > 0) {

        trk->vosLen = enc->extradata_size;

        trk->vosData = av_malloc(trk->vosLen);

        memcpy(trk->vosData, enc->extradata, trk->vosLen);

    }



    if (enc->codec_id == CODEC_ID_H264 && trk->vosLen > 0 && *(uint8_t *)trk->vosData != 1) {

        /* from x264 or from bytestream h264 */

        /* nal reformating needed */

        int ret = ff_avc_parse_nal_units(pkt->data, &pkt->data, &pkt->size);

        if (ret < 0)

            return ret;

        assert(pkt->size);

        size = pkt->size;

    } else if (enc->codec_id == CODEC_ID_DNXHD && !trk->vosLen) {

        /* copy frame to create needed atoms */

        trk->vosLen = size;

        trk->vosData = av_malloc(size);



        memcpy(trk->vosData, pkt->data, size);

    }



    if (!(trk->entry % MOV_INDEX_CLUSTER_SIZE)) {

        trk->cluster = av_realloc(trk->cluster, (trk->entry + MOV_INDEX_CLUSTER_SIZE) * sizeof(*trk->cluster));

        if (!trk->cluster)

            return -1;

    }



    trk->cluster[trk->entry].pos = url_ftell(pb);

    trk->cluster[trk->entry].samplesInChunk = samplesInChunk;

    trk->cluster[trk->entry].size = size;

    trk->cluster[trk->entry].entries = samplesInChunk;

    trk->cluster[trk->entry].dts = pkt->dts;

    trk->trackDuration = pkt->dts - trk->cluster[0].dts + pkt->duration;



    if (pkt->pts == AV_NOPTS_VALUE) {

        av_log(s, AV_LOG_WARNING, "pts has no value\n");

        pkt->pts = pkt->dts;

    }

    if (pkt->dts != pkt->pts)

        trk->hasBframes = 1;

    trk->cluster[trk->entry].cts = pkt->pts - pkt->dts;

    trk->cluster[trk->entry].key_frame = !!(pkt->flags & PKT_FLAG_KEY);

    if(trk->cluster[trk->entry].key_frame)

        trk->hasKeyframes++;

    trk->entry++;

    trk->sampleCount += samplesInChunk;

    mov->mdat_size += size;



    put_buffer(pb, pkt->data, size);



    put_flush_packet(pb);

    return 0;

}