static int xan_decode_frame(AVCodecContext *avctx,

                            void *data, int *data_size,

                            AVPacket *avpkt)

{

    XanContext *s = avctx->priv_data;

    int ftype;

    int ret;



    s->pic.reference = 1;

    s->pic.buffer_hints = FF_BUFFER_HINTS_VALID |

                          FF_BUFFER_HINTS_PRESERVE |

                          FF_BUFFER_HINTS_REUSABLE;

    if ((ret = avctx->reget_buffer(avctx, &s->pic))) {

        av_log(s->avctx, AV_LOG_ERROR, "reget_buffer() failed\n");

        return ret;

    }



    ftype = AV_RL32(avpkt->data);

    switch (ftype) {

    case 0:

        ret = xan_decode_frame_type0(avctx, avpkt);

        break;

    case 1:

        ret = xan_decode_frame_type1(avctx, avpkt);

        break;

    default:

        av_log(avctx, AV_LOG_ERROR, "Unknown frame type %d\n", ftype);

        return -1;

    }

    if (ret)

        return ret;



    *data_size = sizeof(AVFrame);

    *(AVFrame*)data = s->pic;



    return avpkt->size;

}
