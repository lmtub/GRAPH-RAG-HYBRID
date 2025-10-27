static int vp8_handle_packet(AVFormatContext *ctx, PayloadContext *vp8,

                             AVStream *st, AVPacket *pkt, uint32_t *timestamp,

                             const uint8_t *buf, int len, uint16_t seq,

                             int flags)

{

    int start_partition, end_packet;

    int extended_bits, part_id;

    int pictureid_present = 0, tl0picidx_present = 0, tid_present = 0,

        keyidx_present = 0;

    int pictureid = -1, pictureid_mask = 0;

    int returned_old_frame = 0;

    uint32_t old_timestamp;



    if (!buf) {

        if (vp8->data) {

            int ret = ff_rtp_finalize_packet(pkt, &vp8->data, st->index);

            if (ret < 0)

                return ret;

            *timestamp = vp8->timestamp;

            if (vp8->sequence_dirty)

                pkt->flags |= AV_PKT_FLAG_CORRUPT;

            return 0;

        }

        return AVERROR(EAGAIN);

    }



    if (len < 1)

        return AVERROR_INVALIDDATA;



    extended_bits   = buf[0] & 0x80;

    start_partition = buf[0] & 0x10;

    part_id         = buf[0] & 0x0f;

    end_packet      = flags & RTP_FLAG_MARKER;

    buf++;

    len--;

    if (extended_bits) {

        if (len < 1)

            return AVERROR_INVALIDDATA;

        pictureid_present = buf[0] & 0x80;

        tl0picidx_present = buf[0] & 0x40;

        tid_present       = buf[0] & 0x20;

        keyidx_present    = buf[0] & 0x10;

        buf++;

        len--;

    }

    if (pictureid_present) {

        if (len < 1)

            return AVERROR_INVALIDDATA;

        if (buf[0] & 0x80) {

            if (len < 2)

                return AVERROR_INVALIDDATA;

            pictureid = AV_RB16(buf) & 0x7fff;

            pictureid_mask = 0x7fff;

            buf += 2;

            len -= 2;

        } else {

            pictureid = buf[0] & 0x7f;

            pictureid_mask = 0x7f;

            buf++;

            len--;

        }

    }

    if (tl0picidx_present) {

        // Ignoring temporal level zero index

        buf++;

        len--;

    }

    if (tid_present || keyidx_present) {

        // Ignoring temporal layer index, layer sync bit and keyframe index

        buf++;

        len--;

    }

    if (len < 1)

        return AVERROR_INVALIDDATA;



    if (start_partition && part_id == 0 && len >= 3) {

        int res;

        int non_key = buf[0] & 0x01;

        if (!non_key) {

            vp8_free_buffer(vp8);

            // Keyframe, decoding ok again

            vp8->sequence_ok = 1;

            vp8->sequence_dirty = 0;

            vp8->got_keyframe = 1;

        } else {

            int can_continue = vp8->data && !vp8->is_keyframe &&

                               avio_tell(vp8->data) >= vp8->first_part_size;

            if (!vp8->sequence_ok)

                return AVERROR(EAGAIN);

            if (!vp8->got_keyframe)

                return vp8_broken_sequence(ctx, vp8, "Keyframe missing\n");

            if (pictureid >= 0) {

                if (pictureid != ((vp8->prev_pictureid + 1) & pictureid_mask)) {

                    return vp8_broken_sequence(ctx, vp8,

                                               "Missed a picture, sequence broken\n");

                } else {

                    if (vp8->data && !can_continue)

                        return vp8_broken_sequence(ctx, vp8,

                                                   "Missed a picture, sequence broken\n");

                }

            } else {

                uint16_t expected_seq = vp8->prev_seq + 1;

                int16_t diff = seq - expected_seq;

                if (vp8->data) {

                    // No picture id, so we can't know if missed packets

                    // contained any new frames. If diff == 0, we did get

                    // later packets from the same frame (matching timestamp),

                    // so we know we didn't miss any frame. If diff == 1 and

                    // we still have data (not flushed by the end of frame

                    // marker), the single missed packet must have been part

                    // of the same frame.

                    if ((diff == 0 || diff == 1) && can_continue) {

                        // Proceed with what we have

                    } else {

                        return vp8_broken_sequence(ctx, vp8,

                                                   "Missed too much, sequence broken\n");

                    }

                } else {

                    if (diff != 0)

                        return vp8_broken_sequence(ctx, vp8,

                                                   "Missed unknown data, sequence broken\n");

                }

            }

            if (vp8->data) {

                vp8->sequence_dirty = 1;

                if (avio_tell(vp8->data) >= vp8->first_part_size) {

                    int ret = ff_rtp_finalize_packet(pkt, &vp8->data, st->index);

                    if (ret < 0)

                        return ret;

                    pkt->flags |= AV_PKT_FLAG_CORRUPT;

                    returned_old_frame = 1;

                    old_timestamp = vp8->timestamp;

                } else {

                    // Shouldn't happen

                    vp8_free_buffer(vp8);

                }

            }

        }

        vp8->first_part_size = (AV_RL16(&buf[1]) << 3 | buf[0] >> 5) + 3;

        if ((res = avio_open_dyn_buf(&vp8->data)) < 0)

            return res;

        vp8->timestamp = *timestamp;

        vp8->broken_frame = 0;

        vp8->prev_pictureid = pictureid;

        vp8->is_keyframe = !non_key;

    } else {

        uint16_t expected_seq = vp8->prev_seq + 1;



        if (!vp8->sequence_ok)

            return AVERROR(EAGAIN);



        if (vp8->timestamp != *timestamp) {

            // Missed the start of the new frame, sequence broken

            return vp8_broken_sequence(ctx, vp8,

                                       "Received no start marker; dropping frame\n");

        }



        if (seq != expected_seq) {

            if (vp8->is_keyframe) {

                return vp8_broken_sequence(ctx, vp8,

                                           "Missed part of a keyframe, sequence broken\n");

            } else if (vp8->data && avio_tell(vp8->data) >= vp8->first_part_size) {

                vp8->broken_frame = 1;

                vp8->sequence_dirty = 1;

            } else {

                return vp8_broken_sequence(ctx, vp8,

                                           "Missed part of the first partition, sequence broken\n");

            }

        }

    }



    if (!vp8->data)

        return vp8_broken_sequence(ctx, vp8, "Received no start marker\n");



    vp8->prev_seq = seq;

    if (!vp8->broken_frame)

        avio_write(vp8->data, buf, len);



    if (returned_old_frame) {

        *timestamp = old_timestamp;

        return end_packet ? 1 : 0;

    }



    if (end_packet) {

        int ret;

        ret = ff_rtp_finalize_packet(pkt, &vp8->data, st->index);

        if (ret < 0)

            return ret;

        if (vp8->sequence_dirty)

            pkt->flags |= AV_PKT_FLAG_CORRUPT;

        return 0;

    }



    return AVERROR(EAGAIN);

}
