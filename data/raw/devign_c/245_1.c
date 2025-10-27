static int cbs_h265_read_nal_unit(CodedBitstreamContext *ctx,

                                  CodedBitstreamUnit *unit)

{

    BitstreamContext bc;

    int err;



    err = bitstream_init(&bc, unit->data, 8 * unit->data_size);

    if (err < 0)

        return err;



    switch (unit->type) {

    case HEVC_NAL_VPS:

        {

            H265RawVPS *vps;



            vps = av_mallocz(sizeof(*vps));

            if (!vps)

                return AVERROR(ENOMEM);

            err = cbs_h265_read_vps(ctx, &bc, vps);

            if (err >= 0)

                err = cbs_h265_replace_vps(ctx, vps);

            if (err < 0) {

                av_free(vps);

                return err;

            }



            unit->content = vps;

        }

        break;

    case HEVC_NAL_SPS:

        {

            H265RawSPS *sps;



            sps = av_mallocz(sizeof(*sps));

            if (!sps)

                return AVERROR(ENOMEM);

            err = cbs_h265_read_sps(ctx, &bc, sps);

            if (err >= 0)

                err = cbs_h265_replace_sps(ctx, sps);

            if (err < 0) {

                av_free(sps);

                return err;

            }



            unit->content = sps;

        }

        break;



    case HEVC_NAL_PPS:

        {

            H265RawPPS *pps;



            pps = av_mallocz(sizeof(*pps));

            if (!pps)

                return AVERROR(ENOMEM);

            err = cbs_h265_read_pps(ctx, &bc, pps);

            if (err >= 0)

                err = cbs_h265_replace_pps(ctx, pps);

            if (err < 0) {

                av_free(pps);

                return err;

            }



            unit->content = pps;

        }

        break;



    case HEVC_NAL_TRAIL_N:

    case HEVC_NAL_TRAIL_R:

    case HEVC_NAL_TSA_N:

    case HEVC_NAL_TSA_R:

    case HEVC_NAL_STSA_N:

    case HEVC_NAL_STSA_R:

    case HEVC_NAL_RADL_N:

    case HEVC_NAL_RADL_R:

    case HEVC_NAL_RASL_N:

    case HEVC_NAL_RASL_R:

    case HEVC_NAL_BLA_W_LP:

    case HEVC_NAL_BLA_W_RADL:

    case HEVC_NAL_BLA_N_LP:

    case HEVC_NAL_IDR_W_RADL:

    case HEVC_NAL_IDR_N_LP:

    case HEVC_NAL_CRA_NUT:

        {

            H265RawSlice *slice;

            int pos, len;



            slice = av_mallocz(sizeof(*slice));

            if (!slice)

                return AVERROR(ENOMEM);

            err = cbs_h265_read_slice_segment_header(ctx, &bc, &slice->header);

            if (err < 0) {

                av_free(slice);

                return err;

            }



            pos = bitstream_tell(&bc);

            len = unit->data_size;

            if (!unit->data[len - 1]) {

                int z;

                for (z = 0; z < len && !unit->data[len - z - 1]; z++);

                av_log(ctx->log_ctx, AV_LOG_DEBUG, "Deleted %d trailing zeroes "

                       "from slice data.\n", z);

                len -= z;

            }



            slice->data_size = len - pos / 8;

            slice->data = av_malloc(slice->data_size);

            if (!slice->data) {

                av_free(slice);

                return AVERROR(ENOMEM);

            }

            memcpy(slice->data,

                   unit->data + pos / 8, slice->data_size);

            slice->data_bit_start = pos % 8;



            unit->content = slice;

        }

        break;



    case HEVC_NAL_AUD:

        {

            H265RawAUD *aud;



            aud = av_mallocz(sizeof(*aud));

            if (!aud)

                return AVERROR(ENOMEM);

            err = cbs_h265_read_aud(ctx, &bc, aud);

            if (err < 0) {

                av_free(aud);

                return err;

            }



            unit->content = aud;

        }

        break;



    default:

        return AVERROR(ENOSYS);

    }



    return 0;

}
