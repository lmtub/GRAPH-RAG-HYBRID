static int mxf_read_local_tags(MXFContext *mxf, KLVPacket *klv, int (*read_child)(), int ctx_size, enum MXFMetadataSetType type)

{

    ByteIOContext *pb = mxf->fc->pb;

    MXFMetadataSet *ctx = ctx_size ? av_mallocz(ctx_size) : mxf;

    uint64_t klv_end = url_ftell(pb) + klv->length;



    if (!ctx)

        return -1;

    while (url_ftell(pb) + 4 < klv_end) {

        int tag = get_be16(pb);

        int size = get_be16(pb); /* KLV specified by 0x53 */

        uint64_t next = url_ftell(pb) + size;

        UID uid = {0};



        dprintf(mxf->fc, "local tag %#04x size %d\n", tag, size);

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

                    dprintf(mxf->fc, "local tag %#04x\n", local_tag);

                    PRINT_KEY(mxf->fc, "uid", uid);

                }

            }

        }

        if (ctx_size && tag == 0x3C0A)

            get_buffer(pb, ctx->uid, 16);

        else if (read_child(ctx, pb, tag, size, uid) < 0)

            return -1;



        url_fseek(pb, next, SEEK_SET);

    }

    if (ctx_size) ctx->type = type;

    return ctx_size ? mxf_add_metadata_set(mxf, ctx) : 0;

}
