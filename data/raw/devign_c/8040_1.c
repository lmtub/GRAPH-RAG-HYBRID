static int mxf_read_packet_old(AVFormatContext *s, AVPacket *pkt)

{

    KLVPacket klv;

    MXFContext *mxf = s->priv_data;

    int ret;



    while ((ret = klv_read_packet(&klv, s->pb)) == 0) {

        PRINT_KEY(s, "read packet", klv.key);

        av_log(s, AV_LOG_TRACE, "size %"PRIu64" offset %#"PRIx64"\n", klv.length, klv.offset);

        if (IS_KLV_KEY(klv.key, mxf_encrypted_triplet_key)) {

            ret = mxf_decrypt_triplet(s, pkt, &klv);

            if (ret < 0) {

                av_log(s, AV_LOG_ERROR, "invalid encoded triplet\n");

                return ret;

            }

            return 0;

        }

        if (IS_KLV_KEY(klv.key, mxf_essence_element_key) ||

            IS_KLV_KEY(klv.key, mxf_canopus_essence_element_key) ||

            IS_KLV_KEY(klv.key, mxf_avid_essence_element_key)) {

            int index = mxf_get_stream_index(s, &klv);

            int64_t next_ofs, next_klv;

            AVStream *st;

            MXFTrack *track;

            AVCodecParameters *par;



            if (index < 0) {

                av_log(s, AV_LOG_ERROR,

                       "error getting stream index %"PRIu32"\n",

                       AV_RB32(klv.key + 12));

                goto skip;

            }



            st = s->streams[index];

            track = st->priv_data;



            if (s->streams[index]->discard == AVDISCARD_ALL)

                goto skip;



            next_klv = avio_tell(s->pb) + klv.length;

            next_ofs = mxf_set_current_edit_unit(mxf, klv.offset);



            if (next_ofs >= 0 && next_klv > next_ofs) {

                /* if this check is hit then it's possible OPAtom was treated as OP1a

                 * truncate the packet since it's probably very large (>2 GiB is common) */

                avpriv_request_sample(s,

                                      "OPAtom misinterpreted as OP1a? "

                                      "KLV for edit unit %i extending into "

                                      "next edit unit",

                                      mxf->current_edit_unit);

                klv.length = next_ofs - avio_tell(s->pb);

            }



            /* check for 8 channels AES3 element */

            if (klv.key[12] == 0x06 && klv.key[13] == 0x01 && klv.key[14] == 0x10) {

                ret = mxf_get_d10_aes3_packet(s->pb, s->streams[index],

                                              pkt, klv.length);

                if (ret < 0) {

                    av_log(s, AV_LOG_ERROR, "error reading D-10 aes3 frame\n");

                    return ret;

                }

            } else {

                ret = av_get_packet(s->pb, pkt, klv.length);

                if (ret < 0)

                    return ret;

            }

            pkt->stream_index = index;

            pkt->pos = klv.offset;



            par = st->codecpar;



            if (par->codec_type == AVMEDIA_TYPE_VIDEO && next_ofs >= 0) {

                /* mxf->current_edit_unit good - see if we have an

                 * index table to derive timestamps from */

                MXFIndexTable *t = &mxf->index_tables[0];



                if (mxf->nb_index_tables >= 1 && mxf->current_edit_unit < t->nb_ptses) {

                    pkt->dts = mxf->current_edit_unit + t->first_dts;

                    pkt->pts = t->ptses[mxf->current_edit_unit];

                } else if (track->intra_only) {

                    /* intra-only -> PTS = EditUnit.

                     * let utils.c figure out DTS since it can be < PTS if low_delay = 0 (Sony IMX30) */

                    pkt->pts = mxf->current_edit_unit;

                }

            } else if (par->codec_type == AVMEDIA_TYPE_AUDIO) {

                ret = mxf_set_audio_pts(mxf, par, pkt);

                if (ret < 0)

                    return ret;

            }



            /* seek for truncated packets */

            avio_seek(s->pb, next_klv, SEEK_SET);



            return 0;

        } else

        skip:

            avio_skip(s->pb, klv.length);

    }

    return avio_feof(s->pb) ? AVERROR_EOF : ret;

}
