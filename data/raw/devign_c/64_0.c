static int qcow2_read_extensions(BlockDriverState *bs, uint64_t start_offset,

                                 uint64_t end_offset, void **p_feature_table,

                                 int flags, Error **errp)

{

    BDRVQcow2State *s = bs->opaque;

    QCowExtension ext;

    uint64_t offset;

    int ret;



#ifdef DEBUG_EXT

    printf("qcow2_read_extensions: start=%ld end=%ld\n", start_offset, end_offset);

#endif

    offset = start_offset;

    while (offset < end_offset) {



#ifdef DEBUG_EXT

        /* Sanity check */

        if (offset > s->cluster_size)

            printf("qcow2_read_extension: suspicious offset %lu\n", offset);



        printf("attempting to read extended header in offset %lu\n", offset);

#endif



        ret = bdrv_pread(bs->file, offset, &ext, sizeof(ext));

        if (ret < 0) {

            error_setg_errno(errp, -ret, "qcow2_read_extension: ERROR: "

                             "pread fail from offset %" PRIu64, offset);

            return 1;

        }

        be32_to_cpus(&ext.magic);

        be32_to_cpus(&ext.len);

        offset += sizeof(ext);

#ifdef DEBUG_EXT

        printf("ext.magic = 0x%x\n", ext.magic);

#endif

        if (offset > end_offset || ext.len > end_offset - offset) {

            error_setg(errp, "Header extension too large");

            return -EINVAL;

        }



        switch (ext.magic) {

        case QCOW2_EXT_MAGIC_END:

            return 0;



        case QCOW2_EXT_MAGIC_BACKING_FORMAT:

            if (ext.len >= sizeof(bs->backing_format)) {

                error_setg(errp, "ERROR: ext_backing_format: len=%" PRIu32

                           " too large (>=%zu)", ext.len,

                           sizeof(bs->backing_format));

                return 2;

            }

            ret = bdrv_pread(bs->file, offset, bs->backing_format, ext.len);

            if (ret < 0) {

                error_setg_errno(errp, -ret, "ERROR: ext_backing_format: "

                                 "Could not read format name");

                return 3;

            }

            bs->backing_format[ext.len] = '\0';

            s->image_backing_format = g_strdup(bs->backing_format);

#ifdef DEBUG_EXT

            printf("Qcow2: Got format extension %s\n", bs->backing_format);

#endif

            break;



        case QCOW2_EXT_MAGIC_FEATURE_TABLE:

            if (p_feature_table != NULL) {

                void* feature_table = g_malloc0(ext.len + 2 * sizeof(Qcow2Feature));

                ret = bdrv_pread(bs->file, offset , feature_table, ext.len);

                if (ret < 0) {

                    error_setg_errno(errp, -ret, "ERROR: ext_feature_table: "

                                     "Could not read table");

                    return ret;

                }



                *p_feature_table = feature_table;

            }

            break;



        case QCOW2_EXT_MAGIC_CRYPTO_HEADER: {

            unsigned int cflags = 0;

            if (s->crypt_method_header != QCOW_CRYPT_LUKS) {

                error_setg(errp, "CRYPTO header extension only "

                           "expected with LUKS encryption method");

                return -EINVAL;

            }

            if (ext.len != sizeof(Qcow2CryptoHeaderExtension)) {

                error_setg(errp, "CRYPTO header extension size %u, "

                           "but expected size %zu", ext.len,

                           sizeof(Qcow2CryptoHeaderExtension));

                return -EINVAL;

            }



            ret = bdrv_pread(bs->file, offset, &s->crypto_header, ext.len);

            if (ret < 0) {

                error_setg_errno(errp, -ret,

                                 "Unable to read CRYPTO header extension");

                return ret;

            }

            be64_to_cpus(&s->crypto_header.offset);

            be64_to_cpus(&s->crypto_header.length);



            if ((s->crypto_header.offset % s->cluster_size) != 0) {

                error_setg(errp, "Encryption header offset '%" PRIu64 "' is "

                           "not a multiple of cluster size '%u'",

                           s->crypto_header.offset, s->cluster_size);

                return -EINVAL;

            }



            if (flags & BDRV_O_NO_IO) {

                cflags |= QCRYPTO_BLOCK_OPEN_NO_IO;

            }

            s->crypto = qcrypto_block_open(s->crypto_opts, "encrypt.",

                                           qcow2_crypto_hdr_read_func,

                                           bs, cflags, errp);

            if (!s->crypto) {

                return -EINVAL;

            }

        }   break;



        default:

            /* unknown magic - save it in case we need to rewrite the header */

            {

                Qcow2UnknownHeaderExtension *uext;



                uext = g_malloc0(sizeof(*uext)  + ext.len);

                uext->magic = ext.magic;

                uext->len = ext.len;

                QLIST_INSERT_HEAD(&s->unknown_header_ext, uext, next);



                ret = bdrv_pread(bs->file, offset , uext->data, uext->len);

                if (ret < 0) {

                    error_setg_errno(errp, -ret, "ERROR: unknown extension: "

                                     "Could not read data");

                    return ret;

                }

            }

            break;

        }



        offset += ((ext.len + 7) & ~7);

    }



    return 0;

}
