static MMSSCPacketType get_tcp_server_response(MMSContext *mms)

{

    int read_result;

    MMSSCPacketType packet_type= -1;



    for(;;) {

        if((read_result= url_read_complete(mms->mms_hd, mms->in_buffer, 8))==8) {

            // handle command packet.

            if(AV_RL32(mms->in_buffer + 4)==0xb00bface) {

                mms->incoming_flags= mms->in_buffer[3];

                read_result= url_read_complete(mms->mms_hd, mms->in_buffer+8, 4);

                if(read_result == 4) {

                    int length_remaining= AV_RL32(mms->in_buffer+8) + 4;

                    int hr;



                    dprintf(NULL, "Length remaining is %d\n", length_remaining);

                    // read the rest of the packet.

                    if (length_remaining < 0

                        || length_remaining > sizeof(mms->in_buffer) - 12) {

                        dprintf(NULL, "Incoming message len %d exceeds buffer len %d\n",

                            length_remaining, sizeof(mms->in_buffer) - 12);



                    read_result = url_read_complete(mms->mms_hd, mms->in_buffer + 12,

                                                  length_remaining) ;

                    if (read_result == length_remaining) {

                        packet_type= AV_RL16(mms->in_buffer+36);

                    } else {

                        dprintf(NULL, "read for packet type failed%d!\n", read_result);








                } else {

                    dprintf(NULL, "read for length remaining failed%d!\n", read_result);



            } else {

                int length_remaining;

                int packet_id_type;

                int tmp;



                assert(mms->remaining_in_len==0);



                // note we cache the first 8 bytes,

                // then fill up the buffer with the others

                tmp                       = AV_RL16(mms->in_buffer + 6);

                length_remaining          = (tmp - 8) & 0xffff;

                mms->incoming_packet_seq  = AV_RL32(mms->in_buffer);

                packet_id_type            = mms->in_buffer[4];

                mms->incoming_flags       = mms->in_buffer[5];



                if (length_remaining < 0

                        || length_remaining > sizeof(mms->in_buffer) - 8) {

                    dprintf(NULL, "Incoming data len %d exceeds buffer len %d\n",

                            length_remaining, sizeof(mms->in_buffer));



                mms->remaining_in_len    = length_remaining;

                mms->read_in_ptr         = mms->in_buffer;

                read_result= url_read_complete(mms->mms_hd, mms->in_buffer, length_remaining);

                if(read_result != length_remaining) {

                    dprintf(NULL, "read_bytes result: %d asking for %d\n",

                            read_result, length_remaining);


                } else {

                    // if we successfully read everything.

                    if(packet_id_type == mms->header_packet_id) {

                        packet_type = SC_PKT_ASF_HEADER;

                        // Store the asf header

                        if(!mms->header_parsed) {

                            void *p = av_realloc(mms->asf_header,

                                              mms->asf_header_size

                                              + mms->remaining_in_len);

                            if (!p) {

                                av_freep(&mms->asf_header);

                                return AVERROR(ENOMEM);


                            mms->asf_header = p;

                            memcpy(mms->asf_header + mms->asf_header_size,

                                                 mms->read_in_ptr,

                                                 mms->remaining_in_len);

                            mms->asf_header_size += mms->remaining_in_len;


                    } else if(packet_id_type == mms->packet_id) {

                        packet_type = SC_PKT_ASF_MEDIA;

                    } else {

                        dprintf(NULL, "packet id type %d is old.", packet_id_type);

                        continue;






            // preprocess some packet type

            if(packet_type == SC_PKT_KEEPALIVE) {

                send_keepalive_packet(mms);

                continue;

            } else if(packet_type == SC_PKT_STREAM_CHANGING) {

                handle_packet_stream_changing_type(mms);

            } else if(packet_type == SC_PKT_ASF_MEDIA) {

                pad_media_packet(mms);


            return packet_type;

        } else {

            if(read_result<0) {

                dprintf(NULL, "Read error (or cancelled) returned %d!\n", read_result);

                packet_type = SC_PKT_CANCEL;

            } else {

                dprintf(NULL, "Read result of zero?!\n");

                packet_type = SC_PKT_NO_DATA;


            return packet_type;


