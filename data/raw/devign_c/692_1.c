static int mxg_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    int ret;

    unsigned int size;

    uint8_t *startmarker_ptr, *end, *search_end, marker;

    MXGContext *mxg = s->priv_data;



    while (!avio_feof(s->pb) && !s->pb->error){

        if (mxg->cache_size <= OVERREAD_SIZE) {

            /* update internal buffer */

            ret = mxg_update_cache(s, DEFAULT_PACKET_SIZE + OVERREAD_SIZE);

            if (ret < 0)

                return ret;

        }

        end = mxg->buffer_ptr + mxg->cache_size;



        /* find start marker - 0xff */

        if (mxg->cache_size > OVERREAD_SIZE) {

            search_end = end - OVERREAD_SIZE;

            startmarker_ptr = mxg_find_startmarker(mxg->buffer_ptr, search_end);

        } else {

            search_end = end;

            startmarker_ptr = mxg_find_startmarker(mxg->buffer_ptr, search_end);

            if (startmarker_ptr >= search_end - 1 ||

                *(startmarker_ptr + 1) != EOI) break;

        }



        if (startmarker_ptr != search_end) { /* start marker found */

            marker = *(startmarker_ptr + 1);

            mxg->buffer_ptr = startmarker_ptr + 2;

            mxg->cache_size = end - mxg->buffer_ptr;



            if (marker == SOI) {

                mxg->soi_ptr = startmarker_ptr;

            } else if (marker == EOI) {

                if (!mxg->soi_ptr) {

                    av_log(s, AV_LOG_WARNING, "Found EOI before SOI, skipping\n");

                    continue;

                }



                pkt->pts = pkt->dts = mxg->dts;

                pkt->stream_index = 0;

#if FF_API_DESTRUCT_PACKET

FF_DISABLE_DEPRECATION_WARNINGS

                pkt->destruct = NULL;

FF_ENABLE_DEPRECATION_WARNINGS

#endif

                pkt->buf  = NULL;

                pkt->size = mxg->buffer_ptr - mxg->soi_ptr;

                pkt->data = mxg->soi_ptr;



                if (mxg->soi_ptr - mxg->buffer > mxg->cache_size) {

                    if (mxg->cache_size > 0) {

                        memcpy(mxg->buffer, mxg->buffer_ptr, mxg->cache_size);

                    }



                    mxg->buffer_ptr = mxg->buffer;

                }

                mxg->soi_ptr = 0;



                return pkt->size;

            } else if ( (SOF0 <= marker && marker <= SOF15) ||

                        (SOS  <= marker && marker <= COM) ) {

                /* all other markers that start marker segment also contain

                   length value (see specification for JPEG Annex B.1) */

                size = AV_RB16(mxg->buffer_ptr);

                if (size < 2)

                    return AVERROR(EINVAL);



                if (mxg->cache_size < size) {

                    ret = mxg_update_cache(s, size);

                    if (ret < 0)

                        return ret;

                    startmarker_ptr = mxg->buffer_ptr - 2;

                    mxg->cache_size = 0;

                } else {

                    mxg->cache_size -= size;

                }



                mxg->buffer_ptr += size;



                if (marker == APP13 && size >= 16) { /* audio data */

                    /* time (GMT) of first sample in usec since 1970, little-endian */

                    pkt->pts = pkt->dts = AV_RL64(startmarker_ptr + 8);

                    pkt->stream_index = 1;

#if FF_API_DESTRUCT_PACKET

FF_DISABLE_DEPRECATION_WARNINGS

                    pkt->destruct = NULL;

FF_ENABLE_DEPRECATION_WARNINGS

#endif

                    pkt->buf  = NULL;

                    pkt->size = size - 14;

                    pkt->data = startmarker_ptr + 16;



                    if (startmarker_ptr - mxg->buffer > mxg->cache_size) {

                        if (mxg->cache_size > 0) {

                            memcpy(mxg->buffer, mxg->buffer_ptr, mxg->cache_size);

                        }

                        mxg->buffer_ptr = mxg->buffer;

                    }



                    return pkt->size;

                } else if (marker == COM && size >= 18 &&

                           !strncmp(startmarker_ptr + 4, "MXF", 3)) {

                    /* time (GMT) of video frame in usec since 1970, little-endian */

                    mxg->dts = AV_RL64(startmarker_ptr + 12);

                }

            }

        } else {

            /* start marker not found */

            mxg->buffer_ptr = search_end;

            mxg->cache_size = OVERREAD_SIZE;

        }

    }



    return AVERROR_EOF;

}
