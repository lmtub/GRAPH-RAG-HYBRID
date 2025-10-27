static void output_packet(OutputFile *of, AVPacket *pkt, OutputStream *ost)

{

    int ret = 0;



    /* apply the output bitstream filters, if any */

    if (ost->nb_bitstream_filters) {

        int idx;




        ret = av_bsf_send_packet(ost->bsf_ctx[0], pkt);

        if (ret < 0)

            goto finish;



        idx = 1;

        while (idx) {

            /* get a packet from the previous filter up the chain */

            ret = av_bsf_receive_packet(ost->bsf_ctx[idx - 1], pkt);

            /* HACK! - aac_adtstoasc updates extradata after filtering the first frame when

             * the api states this shouldn't happen after init(). Propagate it here to the

             * muxer and to the next filters in the chain to workaround this.

             * TODO/FIXME - Make aac_adtstoasc use new packet side data instead of changing

             * par_out->extradata and adapt muxers accordingly to get rid of this. */

            if (!(ost->bsf_extradata_updated[idx - 1] & 1)) {

                ret = avcodec_parameters_copy(ost->st->codecpar, ost->bsf_ctx[idx - 1]->par_out);

                if (ret < 0)

                    goto finish;

                ost->bsf_extradata_updated[idx - 1] |= 1;

            }

            if (ret == AVERROR(EAGAIN)) {

                ret = 0;

                idx--;

                continue;

            } else if (ret < 0)

                goto finish;



            /* send it to the next filter down the chain or to the muxer */

            if (idx < ost->nb_bitstream_filters) {

                /* HACK/FIXME! - See above */

                if (!(ost->bsf_extradata_updated[idx] & 2)) {

                    ret = avcodec_parameters_copy(ost->bsf_ctx[idx]->par_out, ost->bsf_ctx[idx - 1]->par_out);

                    if (ret < 0)

                        goto finish;

                    ost->bsf_extradata_updated[idx] |= 2;

                }

                ret = av_bsf_send_packet(ost->bsf_ctx[idx], pkt);

                if (ret < 0)

                    goto finish;

                idx++;

            } else

                write_packet(of, pkt, ost);

        }

    } else

        write_packet(of, pkt, ost);



finish:

    if (ret < 0 && ret != AVERROR_EOF) {

        av_log(NULL, AV_LOG_ERROR, "Error applying bitstream filters to an output "

               "packet for stream #%d:%d.\n", ost->file_index, ost->index);

        if(exit_on_error)

            exit_program(1);

    }

}