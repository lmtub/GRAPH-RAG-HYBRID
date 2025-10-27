static MMSSCPacketType get_tcp_server_response(MMSTContext *mmst)

{

    int read_result;

    MMSSCPacketType packet_type= -1;

    MMSContext *mms = &mmst->mms;

    for(;;) {

        read_result = url_read_complete(mms->mms_hd, mms->in_buffer, 8);

        if (read_result != 8) {

            if(read_result < 0) {

                av_log(NULL, AV_LOG_ERROR,

                       "Error reading packet header: %d (%s)\n",

                       read_result, strerror(read_result));

                packet_type = SC_PKT_CANCEL;

            } else {

                av_log(NULL, AV_LOG_ERROR,

                       "The server closed the connection\n");

                packet_type = SC_PKT_NO_DATA;

            }

            return packet_type;

        }



        // handle command packet.

        if(AV_RL32(mms->in_buffer + 4)==0xb00bface) {

            int length_remaining, hr;



            mmst->incoming_flags= mms->in_buffer[3];

            read_result= url_read_complete(mms->mms_hd, mms->in_buffer+8, 4);

            if(read_result != 4) {

                av_log(NULL, AV_LOG_ERROR,

                       "Reading command packet length failed: %d (%s)\n",

                       read_result,

                       read_result < 0 ? strerror(read_result) :

                           "The server closed the connection");

                return read_result < 0 ? read_result : AVERROR_IO;

            }



            length_remaining= AV_RL32(mms->in_buffer+8) + 4;

            av_dlog(NULL, "Length remaining is %d\n", length_remaining);

            // read the rest of the packet.

            if (length_remaining < 0

                || length_remaining > sizeof(mms->in_buffer) - 12) {

                av_log(NULL, AV_LOG_ERROR,

                       "Incoming packet length %d exceeds bufsize %zu\n",

                       length_remaining, sizeof(mms->in_buffer) - 12);

                return AVERROR_INVALIDDATA;

            }

            read_result = url_read_complete(mms->mms_hd, mms->in_buffer + 12,

                                            length_remaining) ;

            if (read_result != length_remaining) {

                av_log(NULL, AV_LOG_ERROR,

                       "Reading pkt data (length=%d) failed: %d (%s)\n",

                       length_remaining, read_result,

                       read_result < 0 ? strerror(read_result) :

                           "The server closed the connection");

                return read_result < 0 ? read_result : AVERROR_IO;

            }

            packet_type= AV_RL16(mms->in_buffer+36);

            hr = AV_RL32(mms->in_buffer + 40);

            if (hr) {

                av_log(NULL, AV_LOG_ERROR,

                       "Server sent a message with packet type 0x%x and error status code 0x%08x\n", packet_type, hr);

                return AVERROR_UNKNOWN;

            }

        } else {

            int length_remaining;

            int packet_id_type;

            int tmp;



            // note we cache the first 8 bytes,

            // then fill up the buffer with the others

            tmp                       = AV_RL16(mms->in_buffer + 6);

            length_remaining          = (tmp - 8) & 0xffff;

            mmst->incoming_packet_seq = AV_RL32(mms->in_buffer);

            packet_id_type            = mms->in_buffer[4];

            mmst->incoming_flags      = mms->in_buffer[5];



            if (length_remaining < 0

                || length_remaining > sizeof(mms->in_buffer) - 8) {

                av_log(NULL, AV_LOG_ERROR,

                       "Data length %d is invalid or too large (max=%zu)\n",

                       length_remaining, sizeof(mms->in_buffer));

                return AVERROR_INVALIDDATA;

            }

            mms->remaining_in_len    = length_remaining;

            mms->read_in_ptr         = mms->in_buffer;

            read_result= url_read_complete(mms->mms_hd, mms->in_buffer, length_remaining);

            if(read_result != length_remaining) {

                av_log(NULL, AV_LOG_ERROR,

                       "Failed to read packet data of size %d: %d (%s)\n",

                       length_remaining, read_result,

                       read_result < 0 ? strerror(read_result) :

                           "The server closed the connection");

                return read_result < 0 ? read_result : AVERROR_IO;

            }



            // if we successfully read everything.

            if(packet_id_type == mmst->header_packet_id) {

                packet_type = SC_PKT_ASF_HEADER;

                // Store the asf header

                if(!mms->header_parsed) {

                    void *p = av_realloc(mms->asf_header,

                                  mms->asf_header_size + mms->remaining_in_len);

                    if (!p) {

                        av_freep(&mms->asf_header);

                        return AVERROR(ENOMEM);

                    }

                    mms->asf_header = p;

                    memcpy(mms->asf_header + mms->asf_header_size,

                           mms->read_in_ptr, mms->remaining_in_len);

                    mms->asf_header_size += mms->remaining_in_len;

                }

                // 0x04 means asf header is sent in multiple packets.

                if (mmst->incoming_flags == 0x04)

                    continue;

            } else if(packet_id_type == mmst->packet_id) {

                packet_type = SC_PKT_ASF_MEDIA;

            } else {

                av_dlog(NULL, "packet id type %d is old.", packet_id_type);

                continue;

            }

        }



        // preprocess some packet type

        if(packet_type == SC_PKT_KEEPALIVE) {

            send_keepalive_packet(mmst);

            continue;

        } else if(packet_type == SC_PKT_STREAM_CHANGING) {

            handle_packet_stream_changing_type(mmst);

        } else if(packet_type == SC_PKT_ASF_MEDIA) {

            pad_media_packet(mms);

        }

        return packet_type;

    }

}
