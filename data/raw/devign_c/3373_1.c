static int mxf_read_local_tags(MXFContext *mxf, KLVPacket *klv, MXFMetadataReadFunc *read_child, int ctx_size, enum MXFMetadataSetType type)

{

    AVIOContext *pb = mxf->fc->pb;

    MXFMetadataSet *ctx = ctx_size ? av_mallocz(ctx_size) : mxf;

    uint64_t klv_end = avio_tell(pb) + klv->length;



    if (!ctx)

        return AVERROR(ENOMEM);

    mxf_metadataset_init(ctx, type);

    while (avio_tell(pb) + 4 < klv_end && !avio_feof(pb)) {

        int ret;

        int tag = avio_rb16(pb);

        int size = avio_rb16(pb); /* KLV specified by 0x53 */

        uint64_t next = avio_tell(pb) + size;

        UID uid = {0};



        av_dlog(mxf->fc, "local tag %#04x size %d\n", tag, size);

        if (!size) { /* ignore empty tag, needed for some files with empty UMID tag */

            av_log(mxf->fc, AV_LOG_ERROR, "local tag %#04x with 0 size\n", tag);

            continue;

        }

        if (tag > 0x7FFF) { /* dynamic tag */

            int i;

            for (i = 0; i < mxf->local_tags_count; i++) {

                int local_tag = AV_RB16(mxf->local_tags+i*18);

                if (local_tag == tag) {

                    memcpy(uid, mxf->local_tags+i*18+2, 16);

                    av_dlog(mxf->fc, "local tag %#04x\n", local_tag);

                    PRINT_KEY(mxf->fc, "uid", uid);

                }

            }

        }

        if (ctx_size && tag == 0x3C0A)

            avio_read(pb, ctx->uid, 16);

        else if ((ret = read_child(ctx, pb, tag, size, uid, -1)) < 0)

            return ret;



        /* Accept the 64k local set limit being exceeded (Avid). Don't accept

         * it extending past the end of the KLV though (zzuf5.mxf). */

        if (avio_tell(pb) > klv_end) {

            if (ctx_size)

                av_free(ctx);



            av_log(mxf->fc, AV_LOG_ERROR,

                   "local tag %#04x extends past end of local set @ %#"PRIx64"\n",

                   tag, klv->offset);

            return AVERROR_INVALIDDATA;

        } else if (avio_tell(pb) <= next)   /* only seek forward, else this can loop for a long time */

            avio_seek(pb, next, SEEK_SET);

    }

    if (ctx_size) ctx->type = type;

    return ctx_size ? mxf_add_metadata_set(mxf, ctx) : 0;

}
