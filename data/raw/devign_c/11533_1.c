static int qcow_create2(const char *filename, int64_t total_size,

                        const char *backing_file, const char *backing_format,

                        int flags, size_t cluster_size, int prealloc)

{



    int fd, header_size, backing_filename_len, l1_size, i, shift, l2_bits;

    int ref_clusters, backing_format_len = 0;

    int rounded_ext_bf_len = 0;

    QCowHeader header;

    uint64_t tmp, offset;

    QCowCreateState s1, *s = &s1;

    QCowExtension ext_bf = {0, 0};

    int ret;



    memset(s, 0, sizeof(*s));



    fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC | O_BINARY, 0644);

    if (fd < 0)

        return -1;

    memset(&header, 0, sizeof(header));

    header.magic = cpu_to_be32(QCOW_MAGIC);

    header.version = cpu_to_be32(QCOW_VERSION);

    header.size = cpu_to_be64(total_size * 512);

    header_size = sizeof(header);

    backing_filename_len = 0;

    if (backing_file) {

        if (backing_format) {

            ext_bf.magic = QCOW_EXT_MAGIC_BACKING_FORMAT;

            backing_format_len = strlen(backing_format);

            ext_bf.len = backing_format_len;

            rounded_ext_bf_len = (sizeof(ext_bf) + ext_bf.len + 7) & ~7;

            header_size += rounded_ext_bf_len;

        }

        header.backing_file_offset = cpu_to_be64(header_size);

        backing_filename_len = strlen(backing_file);

        header.backing_file_size = cpu_to_be32(backing_filename_len);

        header_size += backing_filename_len;

    }



    /* Cluster size */

    s->cluster_bits = get_bits_from_size(cluster_size);

    if (s->cluster_bits < MIN_CLUSTER_BITS ||

        s->cluster_bits > MAX_CLUSTER_BITS)

    {

        fprintf(stderr, "Cluster size must be a power of two between "

            "%d and %dk\n",

            1 << MIN_CLUSTER_BITS,

            1 << (MAX_CLUSTER_BITS - 10));

        return -EINVAL;

    }

    s->cluster_size = 1 << s->cluster_bits;



    header.cluster_bits = cpu_to_be32(s->cluster_bits);

    header_size = (header_size + 7) & ~7;

    if (flags & BLOCK_FLAG_ENCRYPT) {

        header.crypt_method = cpu_to_be32(QCOW_CRYPT_AES);

    } else {

        header.crypt_method = cpu_to_be32(QCOW_CRYPT_NONE);

    }

    l2_bits = s->cluster_bits - 3;

    shift = s->cluster_bits + l2_bits;

    l1_size = (((total_size * 512) + (1LL << shift) - 1) >> shift);

    offset = align_offset(header_size, s->cluster_size);

    s->l1_table_offset = offset;

    header.l1_table_offset = cpu_to_be64(s->l1_table_offset);

    header.l1_size = cpu_to_be32(l1_size);

    offset += align_offset(l1_size * sizeof(uint64_t), s->cluster_size);



    s->refcount_table = qemu_mallocz(s->cluster_size);



    s->refcount_table_offset = offset;

    header.refcount_table_offset = cpu_to_be64(offset);

    header.refcount_table_clusters = cpu_to_be32(1);

    offset += s->cluster_size;

    s->refcount_block_offset = offset;



    /* count how many refcount blocks needed */

    tmp = offset >> s->cluster_bits;

    ref_clusters = (tmp >> (s->cluster_bits - REFCOUNT_SHIFT)) + 1;

    for (i=0; i < ref_clusters; i++) {

        s->refcount_table[i] = cpu_to_be64(offset);

        offset += s->cluster_size;

    }



    s->refcount_block = qemu_mallocz(ref_clusters * s->cluster_size);



    /* update refcounts */

    qcow2_create_refcount_update(s, 0, header_size);

    qcow2_create_refcount_update(s, s->l1_table_offset,

        l1_size * sizeof(uint64_t));

    qcow2_create_refcount_update(s, s->refcount_table_offset, s->cluster_size);

    qcow2_create_refcount_update(s, s->refcount_block_offset,

        ref_clusters * s->cluster_size);



    /* write all the data */

    ret = qemu_write_full(fd, &header, sizeof(header));

    if (ret != sizeof(header)) {

        ret = -1;

        goto exit;

    }

    if (backing_file) {

        if (backing_format_len) {

            char zero[16];

            int padding = rounded_ext_bf_len - (ext_bf.len + sizeof(ext_bf));



            memset(zero, 0, sizeof(zero));

            cpu_to_be32s(&ext_bf.magic);

            cpu_to_be32s(&ext_bf.len);

            ret = qemu_write_full(fd, &ext_bf, sizeof(ext_bf));

            if (ret != sizeof(ext_bf)) {

                ret = -1;

                goto exit;

            }

            ret = qemu_write_full(fd, backing_format, backing_format_len);

            if (ret != backing_format_len) {

                ret = -1;

                goto exit;

            }

            if (padding > 0) {

                ret = qemu_write_full(fd, zero, padding);

                if (ret != padding) {

                    ret = -1;

                    goto exit;

                }

            }

        }

        ret = qemu_write_full(fd, backing_file, backing_filename_len);

        if (ret != backing_filename_len) {

            ret = -1;

            goto exit;

        }

    }

    lseek(fd, s->l1_table_offset, SEEK_SET);

    tmp = 0;

    for(i = 0;i < l1_size; i++) {

        ret = qemu_write_full(fd, &tmp, sizeof(tmp));

        if (ret != sizeof(tmp)) {

            ret = -1;

            goto exit;

        }

    }

    lseek(fd, s->refcount_table_offset, SEEK_SET);

    ret = qemu_write_full(fd, s->refcount_table, s->cluster_size);

    if (ret != s->cluster_size) {

        ret = -1;

        goto exit;

    }



    lseek(fd, s->refcount_block_offset, SEEK_SET);

    ret = qemu_write_full(fd, s->refcount_block,

		    ref_clusters * s->cluster_size);

    if (ret != s->cluster_size) {

        ret = -1;

        goto exit;

    }



    ret = 0;

exit:

    qemu_free(s->refcount_table);

    qemu_free(s->refcount_block);

    close(fd);



    /* Preallocate metadata */

    if (prealloc) {

        BlockDriverState *bs;

        bs = bdrv_new("");

        bdrv_open(bs, filename, BDRV_O_CACHE_WB | BDRV_O_RDWR);

        preallocate(bs);

        bdrv_close(bs);

    }



    return ret;

}
