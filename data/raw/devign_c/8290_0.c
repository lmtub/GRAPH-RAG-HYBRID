static int mov_write_packet(AVFormatContext *s, int stream_index,

                            const uint8_t *buf, int size, int64_t pts)

{

    MOVContext *mov = s->priv_data;

    ByteIOContext *pb = &s->pb;

    AVCodecContext *enc = &s->streams[stream_index]->codec;

    MOVTrack* trk = &mov->tracks[stream_index];

    int cl, id;

    unsigned int samplesInChunk = 0;



    if (url_is_streamed(&s->pb)) return 0; /* Can't handle that */

    if (!size) return 0; /* Discard 0 sized packets */



    if (enc->codec_type == CODEC_TYPE_VIDEO ) {

        samplesInChunk = 1;

    }

    else if (enc->codec_type == CODEC_TYPE_AUDIO ) {

        if( enc->codec_id == CODEC_ID_AMR_NB) {

            /* We must find out how many AMR blocks there are in one packet */

            static uint16_t packed_size[16] =

                {13, 14, 16, 18, 20, 21, 27, 32, 6, 0, 0, 0, 0, 0, 0, 0};

            int len = 0;



            while (len < size && samplesInChunk < 100) {

                len += packed_size[(buf[len] >> 3) & 0x0F];

                samplesInChunk++;

            }

        }

        else if(enc->codec_id == CODEC_ID_PCM_ALAW) {

            samplesInChunk = size/enc->channels;

        }

        else {

            samplesInChunk = 1;

        }

    }



    if ((enc->codec_id == CODEC_ID_MPEG4 || enc->codec_id == CODEC_ID_AAC)

        && trk->vosLen == 0) {

        assert(enc->extradata_size);



        trk->vosLen = enc->extradata_size;

        trk->vosData = av_malloc(trk->vosLen);

        memcpy(trk->vosData, enc->extradata, trk->vosLen);

    }



    cl = trk->entry / MOV_INDEX_CLUSTER_SIZE;

    id = trk->entry % MOV_INDEX_CLUSTER_SIZE;



    if (trk->ents_allocated <= trk->entry) {

        trk->cluster = av_realloc(trk->cluster, (cl+1)*sizeof(void*)); 

        if (!trk->cluster)

            return -1;

        trk->cluster[cl] = av_malloc(MOV_INDEX_CLUSTER_SIZE*sizeof(MOVIentry));

        if (!trk->cluster[cl])

            return -1;

        trk->ents_allocated += MOV_INDEX_CLUSTER_SIZE;

    }

    if (mov->mdat_written == 0) {

        mov_write_mdat_tag(pb, mov);

        mov->mdat_written = 1;

        mov->time = Timestamp();

    }



    trk->cluster[cl][id].pos = url_ftell(pb) - mov->movi_list;

    trk->cluster[cl][id].samplesInChunk = samplesInChunk;

    trk->cluster[cl][id].size = size;

    trk->cluster[cl][id].entries = samplesInChunk;

    if(enc->codec_type == CODEC_TYPE_VIDEO) {

        trk->cluster[cl][id].key_frame = enc->coded_frame->key_frame;

        if(enc->coded_frame->pict_type == FF_I_TYPE)

            trk->hasKeyframes = 1;

    }

    trk->enc = enc;

    trk->entry++;

    trk->sampleCount += samplesInChunk;

    trk->mdat_size += size;



    put_buffer(pb, buf, size);



    put_flush_packet(pb);

    return 0;

}
