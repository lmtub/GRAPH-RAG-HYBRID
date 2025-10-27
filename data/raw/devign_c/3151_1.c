static int do_qsv_decode(AVCodecContext *avctx, QSVContext *q,

                  AVFrame *frame, int *got_frame,

                  AVPacket *avpkt)

{

    QSVFrame *out_frame;

    mfxFrameSurface1 *insurf;

    mfxFrameSurface1 *outsurf;

    mfxSyncPoint *sync;

    mfxBitstream bs = { { { 0 } } };

    int ret;

    int n_out_frames;

    int buffered = 0;

    int flush    = !avpkt->size || q->reinit_pending;



    if (!q->engine_ready) {

        ret = qsv_decode_init(avctx, q, avpkt);

        if (ret)

            return ret;

    }



    if (!flush) {

        if (av_fifo_size(q->input_fifo)) {

            /* we have got rest of previous packet into buffer */

            if (av_fifo_space(q->input_fifo) < avpkt->size) {

                ret = av_fifo_grow(q->input_fifo, avpkt->size);

                if (ret < 0)

                    return ret;

            }

            av_fifo_generic_write(q->input_fifo, avpkt->data, avpkt->size, NULL);

            bs.Data       = q->input_fifo->rptr;

            bs.DataLength = av_fifo_size(q->input_fifo);

            buffered = 1;

        } else {

            bs.Data       = avpkt->data;

            bs.DataLength = avpkt->size;

        }

        bs.MaxLength  = bs.DataLength;

        bs.TimeStamp  = avpkt->pts;

    }



    sync = av_mallocz(sizeof(*sync));

    if (!sync) {

        av_freep(&sync);

        return AVERROR(ENOMEM);

    }



    while (1) {

        ret = get_surface(avctx, q, &insurf);

        if (ret < 0)

            return ret;

        do {

            ret = MFXVideoDECODE_DecodeFrameAsync(q->session, flush ? NULL : &bs,

                                                  insurf, &outsurf, sync);

            if (ret != MFX_WRN_DEVICE_BUSY)

                break;

            av_usleep(500);

        } while (1);



        if (MFX_WRN_VIDEO_PARAM_CHANGED==ret) {

            /* TODO: handle here minor sequence header changing */

        } else if (MFX_ERR_INCOMPATIBLE_VIDEO_PARAM==ret) {

            av_fifo_reset(q->input_fifo);

            flush = q->reinit_pending = 1;

            continue;

        }



        if (*sync) {

            QSVFrame *out_frame = find_frame(q, outsurf);



            if (!out_frame) {

                av_freep(&sync);

                av_log(avctx, AV_LOG_ERROR,

                       "The returned surface does not correspond to any frame\n");

                return AVERROR_BUG;

            }



            out_frame->queued = 1;

            av_fifo_generic_write(q->async_fifo, &out_frame, sizeof(out_frame), NULL);

            av_fifo_generic_write(q->async_fifo, &sync,      sizeof(sync),      NULL);



            continue;

        } else {

            av_freep(&sync);

        }

        if (MFX_ERR_MORE_SURFACE != ret && ret < 0)

            break;

    }



    /* make sure we do not enter an infinite loop if the SDK

     * did not consume any data and did not return anything */

    if (!*sync && !bs.DataOffset && !flush) {

        av_log(avctx, AV_LOG_WARNING, "A decode call did not consume any data\n");

        bs.DataOffset = avpkt->size;

    }



    if (buffered) {

        qsv_fifo_relocate(q->input_fifo, bs.DataOffset);

    } else if (bs.DataOffset!=avpkt->size) {

        /* some data of packet was not consumed. store it to local buffer */

        av_fifo_generic_write(q->input_fifo, avpkt->data+bs.DataOffset,

                              avpkt->size - bs.DataOffset, NULL);

    }



    if (MFX_ERR_MORE_DATA!=ret && ret < 0) {

        av_freep(&sync);

        av_log(avctx, AV_LOG_ERROR, "Error %d during QSV decoding.\n", ret);

        return ff_qsv_error(ret);

    }

    n_out_frames = av_fifo_size(q->async_fifo) / (sizeof(out_frame)+sizeof(sync));



    if (n_out_frames > q->async_depth || (flush && n_out_frames) ) {

        AVFrame *src_frame;



        av_fifo_generic_read(q->async_fifo, &out_frame, sizeof(out_frame), NULL);

        av_fifo_generic_read(q->async_fifo, &sync,      sizeof(sync),      NULL);

        out_frame->queued = 0;



        do {

            ret = MFXVideoCORE_SyncOperation(q->session, *sync, 1000);

        } while (ret == MFX_WRN_IN_EXECUTION);



        av_freep(&sync);



        src_frame = out_frame->frame;



        ret = av_frame_ref(frame, src_frame);

        if (ret < 0)

            return ret;



        outsurf = out_frame->surface;



        frame->pkt_pts = frame->pts = outsurf->Data.TimeStamp;



        frame->repeat_pict =

            outsurf->Info.PicStruct & MFX_PICSTRUCT_FRAME_TRIPLING ? 4 :

            outsurf->Info.PicStruct & MFX_PICSTRUCT_FRAME_DOUBLING ? 2 :

            outsurf->Info.PicStruct & MFX_PICSTRUCT_FIELD_REPEATED ? 1 : 0;

        frame->top_field_first =

            outsurf->Info.PicStruct & MFX_PICSTRUCT_FIELD_TFF;

        frame->interlaced_frame =

            !(outsurf->Info.PicStruct & MFX_PICSTRUCT_PROGRESSIVE);



        *got_frame = 1;

    }



    return avpkt->size;

}
