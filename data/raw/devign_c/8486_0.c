static int mxf_read_partition_pack(void *arg, AVIOContext *pb, int tag, int size, UID uid, int64_t klv_offset)

{

    MXFContext *mxf = arg;

    MXFPartition *partition;

    UID op;

    uint64_t footer_partition;

    uint32_t nb_essence_containers;

    int err;



    if (mxf->partitions_count+1 >= UINT_MAX / sizeof(*mxf->partitions))

        return AVERROR(ENOMEM);



    if ((err = av_reallocp_array(&mxf->partitions, mxf->partitions_count + 1,

                                 sizeof(*mxf->partitions))) < 0) {

        mxf->partitions_count = 0;

        return err;

    }



    if (mxf->parsing_backward) {

        /* insert the new partition pack in the middle

         * this makes the entries in mxf->partitions sorted by offset */

        memmove(&mxf->partitions[mxf->last_forward_partition+1],

                &mxf->partitions[mxf->last_forward_partition],

                (mxf->partitions_count - mxf->last_forward_partition)*sizeof(*mxf->partitions));

        partition = mxf->current_partition = &mxf->partitions[mxf->last_forward_partition];

    } else {

        mxf->last_forward_partition++;

        partition = mxf->current_partition = &mxf->partitions[mxf->partitions_count];

    }



    memset(partition, 0, sizeof(*partition));

    mxf->partitions_count++;

    partition->pack_length = avio_tell(pb) - klv_offset + size;



    switch(uid[13]) {

    case 2:

        partition->type = Header;

        break;

    case 3:

        partition->type = BodyPartition;

        break;

    case 4:

        partition->type = Footer;

        break;

    default:

        av_log(mxf->fc, AV_LOG_ERROR, "unknown partition type %i\n", uid[13]);

        return AVERROR_INVALIDDATA;

    }



    /* consider both footers to be closed (there is only Footer and CompleteFooter) */

    partition->closed = partition->type == Footer || !(uid[14] & 1);

    partition->complete = uid[14] > 2;

    avio_skip(pb, 4);

    partition->kag_size = avio_rb32(pb);

    partition->this_partition = avio_rb64(pb);

    partition->previous_partition = avio_rb64(pb);

    footer_partition = avio_rb64(pb);

    partition->header_byte_count = avio_rb64(pb);

    partition->index_byte_count = avio_rb64(pb);

    partition->index_sid = avio_rb32(pb);

    avio_skip(pb, 8);

    partition->body_sid = avio_rb32(pb);

    avio_read(pb, op, sizeof(UID));

    nb_essence_containers = avio_rb32(pb);



    /* some files don'thave FooterPartition set in every partition */

    if (footer_partition) {

        if (mxf->footer_partition && mxf->footer_partition != footer_partition) {

            av_log(mxf->fc, AV_LOG_ERROR,

                   "inconsistent FooterPartition value: %"PRIu64" != %"PRIu64"\n",

                   mxf->footer_partition, footer_partition);

        } else {

            mxf->footer_partition = footer_partition;

        }

    }



    av_dlog(mxf->fc,

            "PartitionPack: ThisPartition = 0x%"PRIX64

            ", PreviousPartition = 0x%"PRIX64", "

            "FooterPartition = 0x%"PRIX64", IndexSID = %i, BodySID = %i\n",

            partition->this_partition,

            partition->previous_partition, footer_partition,

            partition->index_sid, partition->body_sid);



    /* sanity check PreviousPartition if set */

    if (partition->previous_partition &&

        mxf->run_in + partition->previous_partition >= klv_offset) {

        av_log(mxf->fc, AV_LOG_ERROR,

               "PreviousPartition points to this partition or forward\n");

        return AVERROR_INVALIDDATA;

    }



    if      (op[12] == 1 && op[13] == 1) mxf->op = OP1a;

    else if (op[12] == 1 && op[13] == 2) mxf->op = OP1b;

    else if (op[12] == 1 && op[13] == 3) mxf->op = OP1c;

    else if (op[12] == 2 && op[13] == 1) mxf->op = OP2a;

    else if (op[12] == 2 && op[13] == 2) mxf->op = OP2b;

    else if (op[12] == 2 && op[13] == 3) mxf->op = OP2c;

    else if (op[12] == 3 && op[13] == 1) mxf->op = OP3a;

    else if (op[12] == 3 && op[13] == 2) mxf->op = OP3b;

    else if (op[12] == 3 && op[13] == 3) mxf->op = OP3c;

    else if (op[12] == 64&& op[13] == 1) mxf->op = OPSonyOpt;

    else if (op[12] == 0x10) {

        /* SMPTE 390m: "There shall be exactly one essence container"

         * The following block deals with files that violate this, namely:

         * 2011_DCPTEST_24FPS.V.mxf - two ECs, OP1a

         * abcdefghiv016f56415e.mxf - zero ECs, OPAtom, output by Avid AirSpeed */

        if (nb_essence_containers != 1) {

            MXFOP op = nb_essence_containers ? OP1a : OPAtom;



            /* only nag once */

            if (!mxf->op)

                av_log(mxf->fc, AV_LOG_WARNING,

                       "\"OPAtom\" with %u ECs - assuming %s\n",

                       nb_essence_containers,

                       op == OP1a ? "OP1a" : "OPAtom");



            mxf->op = op;

        } else

            mxf->op = OPAtom;

    } else {

        av_log(mxf->fc, AV_LOG_ERROR, "unknown operational pattern: %02xh %02xh - guessing OP1a\n", op[12], op[13]);

        mxf->op = OP1a;

    }



    if (partition->kag_size <= 0 || partition->kag_size > (1 << 20)) {

        av_log(mxf->fc, AV_LOG_WARNING, "invalid KAGSize %i - guessing ", partition->kag_size);



        if (mxf->op == OPSonyOpt)

            partition->kag_size = 512;

        else

            partition->kag_size = 1;



        av_log(mxf->fc, AV_LOG_WARNING, "%i\n", partition->kag_size);

    }



    return 0;

}
