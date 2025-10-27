int ff_hevc_output_frame(HEVCContext *s, AVFrame *out, int flush)

{

    do {

        int nb_output = 0;

        int min_poc   = INT_MAX;

        int i, min_idx, ret;



        if (s->sh.no_output_of_prior_pics_flag == 1) {

            for (i = 0; i < FF_ARRAY_ELEMS(s->DPB); i++) {

                HEVCFrame *frame = &s->DPB[i];

                if (!(frame->flags & HEVC_FRAME_FLAG_BUMPING) && frame->poc != s->poc &&

                        frame->sequence == s->seq_output) {

                    ff_hevc_unref_frame(s, frame, HEVC_FRAME_FLAG_OUTPUT);

                }

            }

        }



        for (i = 0; i < FF_ARRAY_ELEMS(s->DPB); i++) {

            HEVCFrame *frame = &s->DPB[i];

            if ((frame->flags & HEVC_FRAME_FLAG_OUTPUT) &&

                frame->sequence == s->seq_output) {

                nb_output++;

                if (frame->poc < min_poc) {

                    min_poc = frame->poc;

                    min_idx = i;

                }

            }

        }



        /* wait for more frames before output */

        if (!flush && s->seq_output == s->seq_decode && s->sps &&

            nb_output <= s->sps->temporal_layer[s->sps->max_sub_layers - 1].num_reorder_pics)

            return 0;



        if (nb_output) {

            HEVCFrame *frame = &s->DPB[min_idx];

            AVFrame *dst = out;

            AVFrame *src = frame->frame;

            const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(src->format);

            int pixel_shift = !!(desc->comp[0].depth_minus1 > 7);



            ret = av_frame_ref(out, src);

            if (frame->flags & HEVC_FRAME_FLAG_BUMPING)

                ff_hevc_unref_frame(s, frame, HEVC_FRAME_FLAG_OUTPUT | HEVC_FRAME_FLAG_BUMPING);

            else

                ff_hevc_unref_frame(s, frame, HEVC_FRAME_FLAG_OUTPUT);

            if (ret < 0)

                return ret;



            for (i = 0; i < 3; i++) {

                int hshift = (i > 0) ? desc->log2_chroma_w : 0;

                int vshift = (i > 0) ? desc->log2_chroma_h : 0;

                int off = ((frame->window.left_offset >> hshift) << pixel_shift) +

                          (frame->window.top_offset   >> vshift) * dst->linesize[i];

                dst->data[i] += off;

            }

            av_log(s->avctx, AV_LOG_DEBUG,

                   "Output frame with POC %d.\n", frame->poc);

            return 1;

        }



        if (s->seq_output != s->seq_decode)

            s->seq_output = (s->seq_output + 1) & 0xff;

        else

            break;

    } while (1);



    return 0;

}
