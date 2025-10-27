static int qcow_create(const char *filename, QemuOpts *opts, Error **errp)

{

    int header_size, backing_filename_len, l1_size, shift, i;

    QCowHeader header;

    uint8_t *tmp;

    int64_t total_size = 0;

    char *backing_file = NULL;

    Error *local_err = NULL;

    int ret;

    BlockBackend *qcow_blk;

    const char *encryptfmt = NULL;

    QDict *options;

    QDict *encryptopts = NULL;

    QCryptoBlockCreateOptions *crypto_opts = NULL;

    QCryptoBlock *crypto = NULL;



    /* Read out options */

    total_size = ROUND_UP(qemu_opt_get_size_del(opts, BLOCK_OPT_SIZE, 0),

                          BDRV_SECTOR_SIZE);

    if (total_size == 0) {

        error_setg(errp, "Image size is too small, cannot be zero length");

        ret = -EINVAL;

        goto cleanup;

    }



    backing_file = qemu_opt_get_del(opts, BLOCK_OPT_BACKING_FILE);

    encryptfmt = qemu_opt_get_del(opts, BLOCK_OPT_ENCRYPT_FORMAT);

    if (encryptfmt) {

        if (qemu_opt_get(opts, BLOCK_OPT_ENCRYPT)) {

            error_setg(errp, "Options " BLOCK_OPT_ENCRYPT " and "

                       BLOCK_OPT_ENCRYPT_FORMAT " are mutually exclusive");

            ret = -EINVAL;

            goto cleanup;

        }

    } else if (qemu_opt_get_bool_del(opts, BLOCK_OPT_ENCRYPT, false)) {

        encryptfmt = "aes";

    }



    ret = bdrv_create_file(filename, opts, &local_err);

    if (ret < 0) {

        error_propagate(errp, local_err);

        goto cleanup;

    }



    qcow_blk = blk_new_open(filename, NULL, NULL,

                            BDRV_O_RDWR | BDRV_O_RESIZE | BDRV_O_PROTOCOL,

                            &local_err);

    if (qcow_blk == NULL) {

        error_propagate(errp, local_err);

        ret = -EIO;

        goto cleanup;

    }



    blk_set_allow_write_beyond_eof(qcow_blk, true);



    ret = blk_truncate(qcow_blk, 0, PREALLOC_MODE_OFF, errp);

    if (ret < 0) {

        goto exit;

    }



    memset(&header, 0, sizeof(header));

    header.magic = cpu_to_be32(QCOW_MAGIC);

    header.version = cpu_to_be32(QCOW_VERSION);

    header.size = cpu_to_be64(total_size);

    header_size = sizeof(header);

    backing_filename_len = 0;

    if (backing_file) {

        if (strcmp(backing_file, "fat:")) {

            header.backing_file_offset = cpu_to_be64(header_size);

            backing_filename_len = strlen(backing_file);

            header.backing_file_size = cpu_to_be32(backing_filename_len);

            header_size += backing_filename_len;

        } else {

            /* special backing file for vvfat */

            g_free(backing_file);

            backing_file = NULL;

        }

        header.cluster_bits = 9; /* 512 byte cluster to avoid copying

                                    unmodified sectors */

        header.l2_bits = 12; /* 32 KB L2 tables */

    } else {

        header.cluster_bits = 12; /* 4 KB clusters */

        header.l2_bits = 9; /* 4 KB L2 tables */

    }

    header_size = (header_size + 7) & ~7;

    shift = header.cluster_bits + header.l2_bits;

    l1_size = (total_size + (1LL << shift) - 1) >> shift;



    header.l1_table_offset = cpu_to_be64(header_size);



    options = qemu_opts_to_qdict(opts, NULL);

    qdict_extract_subqdict(options, &encryptopts, "encrypt.");

    QDECREF(options);

    if (encryptfmt) {

        if (!g_str_equal(encryptfmt, "aes")) {

            error_setg(errp, "Unknown encryption format '%s', expected 'aes'",

                       encryptfmt);

            ret = -EINVAL;

            goto exit;

        }

        header.crypt_method = cpu_to_be32(QCOW_CRYPT_AES);



        crypto_opts = block_crypto_create_opts_init(

            Q_CRYPTO_BLOCK_FORMAT_QCOW, encryptopts, errp);

        if (!crypto_opts) {

            ret = -EINVAL;

            goto exit;

        }



        crypto = qcrypto_block_create(crypto_opts, "encrypt.",

                                      NULL, NULL, NULL, errp);

        if (!crypto) {

            ret = -EINVAL;

            goto exit;

        }

    } else {

        header.crypt_method = cpu_to_be32(QCOW_CRYPT_NONE);

    }



    /* write all the data */

    ret = blk_pwrite(qcow_blk, 0, &header, sizeof(header), 0);

    if (ret != sizeof(header)) {

        goto exit;

    }



    if (backing_file) {

        ret = blk_pwrite(qcow_blk, sizeof(header),

                         backing_file, backing_filename_len, 0);

        if (ret != backing_filename_len) {

            goto exit;

        }

    }



    tmp = g_malloc0(BDRV_SECTOR_SIZE);

    for (i = 0; i < DIV_ROUND_UP(sizeof(uint64_t) * l1_size, BDRV_SECTOR_SIZE);

         i++) {

        ret = blk_pwrite(qcow_blk, header_size + BDRV_SECTOR_SIZE * i,

                         tmp, BDRV_SECTOR_SIZE, 0);

        if (ret != BDRV_SECTOR_SIZE) {

            g_free(tmp);

            goto exit;

        }

    }



    g_free(tmp);

    ret = 0;

exit:

    blk_unref(qcow_blk);

cleanup:

    QDECREF(encryptopts);

    qcrypto_block_free(crypto);

    qapi_free_QCryptoBlockCreateOptions(crypto_opts);

    g_free(backing_file);

    return ret;

}
