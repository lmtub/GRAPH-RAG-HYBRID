static int vorbis_encode_frame(AVCodecContext *avctx, AVPacket *avpkt,

                               const AVFrame *frame, int *got_packet_ptr)

{

    vorbis_enc_context *venc = avctx->priv_data;

    int i, ret, need_more;

    int frame_size = 1 << (venc->log2_blocksize[1] - 1);

    vorbis_enc_mode *mode;

    vorbis_enc_mapping *mapping;

    PutBitContext pb;



    if (frame) {

        if ((ret = ff_af_queue_add(&venc->afq, frame)) < 0)

            return ret;

        ff_bufqueue_add(avctx, &venc->bufqueue, av_frame_clone(frame));

    } else

        if (!venc->afq.remaining_samples)

            return 0;



    need_more = venc->bufqueue.available * avctx->frame_size < frame_size;

    need_more = frame && need_more;

    if (need_more)

        return 0;



    /* Pad the bufqueue with empty frames for encoding the last packet. */

    if (!frame) {

        if (venc->bufqueue.available * avctx->frame_size < frame_size) {

            int frames_needed = (frame_size/avctx->frame_size) - venc->bufqueue.available;

            int i;



            for (i = 0; i < frames_needed; i++) {

               AVFrame *empty = spawn_empty_frame(avctx, venc->channels);

               if (!empty)

                   return AVERROR(ENOMEM);



               ff_bufqueue_add(avctx, &venc->bufqueue, empty);

            }

        }

    }



    move_audio(venc, avctx->frame_size);



    if (!apply_window_and_mdct(venc))

        return 0;



    if ((ret = ff_alloc_packet2(avctx, avpkt, 8192, 0)) < 0)

        return ret;



    init_put_bits(&pb, avpkt->data, avpkt->size);



    if (pb.size_in_bits - put_bits_count(&pb) < 1 + ilog(venc->nmodes - 1)) {

        av_log(avctx, AV_LOG_ERROR, "output buffer is too small\n");

        return AVERROR(EINVAL);

    }



    put_bits(&pb, 1, 0); // magic bit



    put_bits(&pb, ilog(venc->nmodes - 1), 1); // Mode for current frame



    mode    = &venc->modes[1];

    mapping = &venc->mappings[mode->mapping];

    if (mode->blockflag) {

        put_bits(&pb, 1, 1); // Previous windowflag

        put_bits(&pb, 1, 1); // Next windowflag

    }



    for (i = 0; i < venc->channels; i++) {

        vorbis_enc_floor *fc = &venc->floors[mapping->floor[mapping->mux[i]]];

        uint16_t posts[MAX_FLOOR_VALUES];

        floor_fit(venc, fc, &venc->coeffs[i * frame_size], posts, frame_size);

        if (floor_encode(venc, fc, &pb, posts, &venc->floor[i * frame_size], frame_size)) {

            av_log(avctx, AV_LOG_ERROR, "output buffer is too small\n");

            return AVERROR(EINVAL);

        }

    }



    for (i = 0; i < venc->channels * frame_size; i++)

        venc->coeffs[i] /= venc->floor[i];



    for (i = 0; i < mapping->coupling_steps; i++) {

        float *mag = venc->coeffs + mapping->magnitude[i] * frame_size;

        float *ang = venc->coeffs + mapping->angle[i]     * frame_size;

        int j;

        for (j = 0; j < frame_size; j++) {

            float a = ang[j];

            ang[j] -= mag[j];

            if (mag[j] > 0)

                ang[j] = -ang[j];

            if (ang[j] < 0)

                mag[j] = a;

        }

    }



    if (residue_encode(venc, &venc->residues[mapping->residue[mapping->mux[0]]],

                       &pb, venc->coeffs, frame_size, venc->channels)) {

        av_log(avctx, AV_LOG_ERROR, "output buffer is too small\n");

        return AVERROR(EINVAL);

    }



    flush_put_bits(&pb);

    avpkt->size = put_bits_count(&pb) >> 3;



    ff_af_queue_remove(&venc->afq, frame_size, &avpkt->pts, &avpkt->duration);



    if (frame_size > avpkt->duration) {

        uint8_t *side = av_packet_new_side_data(avpkt, AV_PKT_DATA_SKIP_SAMPLES, 10);

        if (!side)

            return AVERROR(ENOMEM);

        AV_WL32(&side[4], frame_size - avpkt->duration);

    }



    *got_packet_ptr = 1;

    return 0;

}
