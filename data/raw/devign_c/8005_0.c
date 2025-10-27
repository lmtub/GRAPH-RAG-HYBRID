static HEVCFrame *find_ref_idx(HEVCContext *s, int poc)

{

    int i;

    int LtMask = (1 << s->sps->log2_max_poc_lsb) - 1;



    for (i = 0; i < FF_ARRAY_ELEMS(s->DPB); i++) {

        HEVCFrame *ref = &s->DPB[i];

        if (ref->frame->buf[0] && (ref->sequence == s->seq_decode)) {

            if ((ref->poc & LtMask) == poc)

                return ref;

        }

    }



    for (i = 0; i < FF_ARRAY_ELEMS(s->DPB); i++) {

        HEVCFrame *ref = &s->DPB[i];

        if (ref->frame->buf[0] && ref->sequence == s->seq_decode) {

            if (ref->poc == poc || (ref->poc & LtMask) == poc)

                return ref;

        }

    }



    av_log(s->avctx, AV_LOG_ERROR,

           "Could not find ref with POC %d\n", poc);

    return NULL;

}
