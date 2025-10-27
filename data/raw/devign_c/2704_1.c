static int h264_handle_packet(AVFormatContext *ctx, PayloadContext *data,

                              AVStream *st, AVPacket *pkt, uint32_t *timestamp,

                              const uint8_t *buf, int len, uint16_t seq,

                              int flags)

{

    uint8_t nal;

    uint8_t type;

    int result = 0;



    if (!len) {

        av_log(ctx, AV_LOG_ERROR, "Empty H264 RTP packet\n");

        return AVERROR_INVALIDDATA;

    }

    nal  = buf[0];

    type = nal & 0x1f;



    assert(data);

    assert(buf);



    /* Simplify the case (these are all the nal types used internally by

     * the h264 codec). */

    if (type >= 1 && type <= 23)

        type = 1;

    switch (type) {

    case 0:                    // undefined, but pass them through

    case 1:

        av_new_packet(pkt, len + sizeof(start_sequence));

        memcpy(pkt->data, start_sequence, sizeof(start_sequence));

        memcpy(pkt->data + sizeof(start_sequence), buf, len);

        COUNT_NAL_TYPE(data, nal);

        break;



    case 24:                   // STAP-A (one packet, multiple nals)

        // consume the STAP-A NAL

        buf++;

        len--;

        // first we are going to figure out the total size

        {

            int pass         = 0;

            int total_length = 0;

            uint8_t *dst     = NULL;



            for (pass = 0; pass < 2; pass++) {

                const uint8_t *src = buf;

                int src_len        = len;



                while (src_len > 2) {

                    uint16_t nal_size = AV_RB16(src);



                    // consume the length of the aggregate

                    src     += 2;

                    src_len -= 2;



                    if (nal_size <= src_len) {

                        if (pass == 0) {

                            // counting

                            total_length += sizeof(start_sequence) + nal_size;

                        } else {

                            // copying

                            assert(dst);

                            memcpy(dst, start_sequence, sizeof(start_sequence));

                            dst += sizeof(start_sequence);

                            memcpy(dst, src, nal_size);

                            COUNT_NAL_TYPE(data, *src);

                            dst += nal_size;

                        }

                    } else {

                        av_log(ctx, AV_LOG_ERROR,

                               "nal size exceeds length: %d %d\n", nal_size, src_len);

                    }



                    // eat what we handled

                    src     += nal_size;

                    src_len -= nal_size;



                    if (src_len < 0)

                        av_log(ctx, AV_LOG_ERROR,

                               "Consumed more bytes than we got! (%d)\n", src_len);

                }



                if (pass == 0) {

                    /* now we know the total size of the packet (with the

                     * start sequences added) */

                    av_new_packet(pkt, total_length);

                    dst = pkt->data;

                } else {

                    assert(dst - pkt->data == total_length);

                }

            }

        }

        break;



    case 25:                   // STAP-B

    case 26:                   // MTAP-16

    case 27:                   // MTAP-24

    case 29:                   // FU-B

        av_log(ctx, AV_LOG_ERROR,

               "Unhandled type (%d) (See RFC for implementation details\n",

               type);

        result = AVERROR(ENOSYS);

        break;



    case 28:                   // FU-A (fragmented nal)

        buf++;

        len--;                 // skip the fu_indicator

        if (len > 1) {

            // these are the same as above, we just redo them here for clarity

            uint8_t fu_indicator      = nal;

            uint8_t fu_header         = *buf;

            uint8_t start_bit         = fu_header >> 7;

            uint8_t av_unused end_bit = (fu_header & 0x40) >> 6;

            uint8_t nal_type          = fu_header & 0x1f;

            uint8_t reconstructed_nal;



            // Reconstruct this packet's true nal; only the data follows.

            /* The original nal forbidden bit and NRI are stored in this

             * packet's nal. */

            reconstructed_nal  = fu_indicator & 0xe0;

            reconstructed_nal |= nal_type;



            // skip the fu_header

            buf++;

            len--;



            if (start_bit)

                COUNT_NAL_TYPE(data, nal_type);

            if (start_bit) {

                /* copy in the start sequence, and the reconstructed nal */

                av_new_packet(pkt, sizeof(start_sequence) + sizeof(nal) + len);

                memcpy(pkt->data, start_sequence, sizeof(start_sequence));

                pkt->data[sizeof(start_sequence)] = reconstructed_nal;

                memcpy(pkt->data + sizeof(start_sequence) + sizeof(nal), buf, len);

            } else {

                av_new_packet(pkt, len);

                memcpy(pkt->data, buf, len);

            }

        } else {

            av_log(ctx, AV_LOG_ERROR, "Too short data for FU-A H264 RTP packet\n");

            result = AVERROR_INVALIDDATA;

        }

        break;



    case 30:                   // undefined

    case 31:                   // undefined

    default:

        av_log(ctx, AV_LOG_ERROR, "Undefined type (%d)\n", type);

        result = AVERROR_INVALIDDATA;

        break;

    }



    pkt->stream_index = st->index;



    return result;

}
