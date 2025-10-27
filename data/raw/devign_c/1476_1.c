static void filter(USPPContext *p, uint8_t *dst[3], uint8_t *src[3],

                   int dst_stride[3], int src_stride[3], int width,

                   int height, uint8_t *qp_store, int qp_stride)

{

    int x, y, i, j;

    const int count = 1<<p->log2_count;



    for (i = 0; i < 3; i++) {

        int is_chroma = !!i;

        int w = width  >> (is_chroma ? p->hsub : 0);

        int h = height >> (is_chroma ? p->vsub : 0);

        int stride = p->temp_stride[i];

        int block = BLOCK >> (is_chroma ? p->hsub : 0);



        if (!src[i] || !dst[i])

            continue;

        for (y = 0; y < h; y++) {

            int index = block + block * stride + y * stride;



            memcpy(p->src[i] + index, src[i] + y * src_stride[i], w );

            for (x = 0; x < block; x++) {

                p->src[i][index     - x - 1] = p->src[i][index +     x    ];

                p->src[i][index + w + x    ] = p->src[i][index + w - x - 1];

            }

        }

        for (y = 0; y < block; y++) {

            memcpy(p->src[i] + (  block-1-y) * stride, p->src[i] + (  y+block  ) * stride, stride);

            memcpy(p->src[i] + (h+block  +y) * stride, p->src[i] + (h-y+block-1) * stride, stride);

        }



        p->frame->linesize[i] = stride;

        memset(p->temp[i], 0, (h + 2 * block) * stride * sizeof(int16_t));

    }



    if (p->qp)

        p->frame->quality = p->qp * FF_QP2LAMBDA;

    else {

        int qpsum=0;

        int qpcount = (height>>4) * (height>>4);



        for (y = 0; y < (height>>4); y++) {

            for (x = 0; x < (width>>4); x++)

                qpsum += qp_store[x + y * qp_stride];

        }

        p->frame->quality = norm_qscale((qpsum + qpcount/2) / qpcount, p->qscale_type) * FF_QP2LAMBDA;

    }

//    init per MB qscale stuff FIXME

    p->frame->height = height;

    p->frame->width  = width;



    for (i = 0; i < count; i++) {

        const int x1 = offset[i+count-1][0];

        const int y1 = offset[i+count-1][1];

        const int x1c = x1 >> p->hsub;

        const int y1c = y1 >> p->vsub;

        const int BLOCKc = BLOCK >> p->hsub;

        int offset;

        AVPacket pkt;

        int got_pkt_ptr;



        av_init_packet(&pkt);

        pkt.data = p->outbuf;

        pkt.size = p->outbuf_size;



        p->frame->data[0] = p->src[0] + x1   + y1   * p->frame->linesize[0];

        p->frame->data[1] = p->src[1] + x1c  + y1c  * p->frame->linesize[1];

        p->frame->data[2] = p->src[2] + x1c  + y1c  * p->frame->linesize[2];

        p->frame->format  = p->avctx_enc[i]->pix_fmt;



        avcodec_encode_video2(p->avctx_enc[i], &pkt, p->frame, &got_pkt_ptr);

        p->frame_dec = p->avctx_enc[i]->coded_frame;



        offset = (BLOCK-x1) + (BLOCK-y1) * p->frame_dec->linesize[0];



        for (y = 0; y < height; y++)

            for (x = 0; x < width; x++)

                p->temp[0][x + y * p->temp_stride[0]] += p->frame_dec->data[0][x + y * p->frame_dec->linesize[0] + offset];



        if (!src[2] || !dst[2])

            continue;



        offset = (BLOCKc-x1c) + (BLOCKc-y1c) * p->frame_dec->linesize[1];



        for (y = 0; y < height>>p->vsub; y++) {

            for (x = 0; x < width>>p->hsub; x++) {

                p->temp[1][x + y * p->temp_stride[1]] += p->frame_dec->data[1][x + y * p->frame_dec->linesize[1] + offset];

                p->temp[2][x + y * p->temp_stride[2]] += p->frame_dec->data[2][x + y * p->frame_dec->linesize[2] + offset];

            }

        }

    }



    for (j = 0; j < 3; j++) {

        int is_chroma = !!j;

        if (!dst[j])

            continue;

        store_slice_c(dst[j], p->temp[j], dst_stride[j], p->temp_stride[j],

                      width  >> (is_chroma ? p->hsub : 0),

                      height >> (is_chroma ? p->vsub : 0),

                      8-p->log2_count);

    }

}
