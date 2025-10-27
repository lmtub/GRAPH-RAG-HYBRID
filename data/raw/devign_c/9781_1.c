static int decode_frame(AVCodecContext *avctx, void *data,

                        int *got_frame, AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size       = avpkt->size;

    H264Context *h     = avctx->priv_data;

    AVFrame *pict      = data;

    int buf_index      = 0;

    Picture *out;

    int i, out_idx;

    int ret;



    h->flags  = avctx->flags;



    /* end of stream, output what is still in the buffers */

    if (buf_size == 0) {

 out:



        h->cur_pic_ptr = NULL;

        h->first_field = 0;



        // FIXME factorize this with the output code below

        out     = h->delayed_pic[0];

        out_idx = 0;

        for (i = 1;

             h->delayed_pic[i] &&

             !h->delayed_pic[i]->f.key_frame &&

             !h->delayed_pic[i]->mmco_reset;

             i++)

            if (h->delayed_pic[i]->poc < out->poc) {

                out     = h->delayed_pic[i];

                out_idx = i;

            }



        for (i = out_idx; h->delayed_pic[i]; i++)

            h->delayed_pic[i] = h->delayed_pic[i + 1];



        if (out) {

            out->reference &= ~DELAYED_PIC_REF;

            ret = output_frame(h, pict, &out->f);

            if (ret < 0)

                return ret;

            *got_frame = 1;

        }



        return buf_index;

    }

    if(h->is_avc && buf_size >= 9 && buf[0]==1 && buf[2]==0 && (buf[4]&0xFC)==0xFC && (buf[5]&0x1F) && buf[8]==0x67){

        int cnt= buf[5]&0x1f;

        const uint8_t *p= buf+6;

        while(cnt--){

            int nalsize= AV_RB16(p) + 2;

            if(nalsize > buf_size - (p-buf) || p[2]!=0x67)

                goto not_extra;

            p += nalsize;

        }

        cnt = *(p++);

        if(!cnt)

            goto not_extra;

        while(cnt--){

            int nalsize= AV_RB16(p) + 2;

            if(nalsize > buf_size - (p-buf) || p[2]!=0x68)

                goto not_extra;

            p += nalsize;

        }



        return ff_h264_decode_extradata(h, buf, buf_size);

    }

not_extra:



    buf_index = decode_nal_units(h, buf, buf_size, 0);

    if (buf_index < 0)

        return -1;



    if (!h->cur_pic_ptr && h->nal_unit_type == NAL_END_SEQUENCE) {

        av_assert0(buf_index <= buf_size);

        goto out;

    }



    if (!(avctx->flags2 & CODEC_FLAG2_CHUNKS) && !h->cur_pic_ptr) {

        if (avctx->skip_frame >= AVDISCARD_NONREF ||

            buf_size >= 4 && !memcmp("Q264", buf, 4))

            return buf_size;

        av_log(avctx, AV_LOG_ERROR, "no frame!\n");

        return -1;

    }



    if (!(avctx->flags2 & CODEC_FLAG2_CHUNKS) ||

        (h->mb_y >= h->mb_height && h->mb_height)) {

        if (avctx->flags2 & CODEC_FLAG2_CHUNKS)

            decode_postinit(h, 1);



        field_end(h, 0);



        /* Wait for second field. */

        *got_frame = 0;

        if (h->next_output_pic && (h->next_output_pic->sync || h->sync>1)) {

            ret = output_frame(h, pict, &h->next_output_pic->f);

            if (ret < 0)

                return ret;

            *got_frame = 1;

            if (CONFIG_MPEGVIDEO) {

                ff_print_debug_info2(h->avctx, h->next_output_pic, pict, h->er.mbskip_table,

                                    &h->low_delay,

                                    h->mb_width, h->mb_height, h->mb_stride, 1);

            }

        }

    }



    assert(pict->data[0] || !*got_frame);



    return get_consumed_bytes(buf_index, buf_size);

}
