static int img_convert(int argc, char **argv)

{

    int c, ret = 0, n, n1, bs_n, bs_i, compress, cluster_size, cluster_sectors;

    int progress = 0, flags;

    const char *fmt, *out_fmt, *cache, *out_baseimg, *out_filename;

    BlockDriver *drv, *proto_drv;

    BlockDriverState **bs = NULL, *out_bs = NULL;

    int64_t total_sectors, nb_sectors, sector_num, bs_offset;

    uint64_t bs_sectors;

    uint8_t * buf = NULL;

    const uint8_t *buf1;

    BlockDriverInfo bdi;

    QEMUOptionParameter *param = NULL, *create_options = NULL;

    QEMUOptionParameter *out_baseimg_param;

    char *options = NULL;

    const char *snapshot_name = NULL;

    float local_progress;

    int min_sparse = 8; /* Need at least 4k of zeros for sparse detection */



    fmt = NULL;

    out_fmt = "raw";

    cache = "unsafe";

    out_baseimg = NULL;

    compress = 0;

    for(;;) {

        c = getopt(argc, argv, "f:O:B:s:hce6o:pS:t:");

        if (c == -1) {

            break;

        }

        switch(c) {

        case '?':

        case 'h':

            help();

            break;

        case 'f':

            fmt = optarg;

            break;

        case 'O':

            out_fmt = optarg;

            break;

        case 'B':

            out_baseimg = optarg;

            break;

        case 'c':

            compress = 1;

            break;

        case 'e':

            error_report("option -e is deprecated, please use \'-o "

                  "encryption\' instead!");

            return 1;

        case '6':

            error_report("option -6 is deprecated, please use \'-o "

                  "compat6\' instead!");

            return 1;

        case 'o':

            options = optarg;

            break;

        case 's':

            snapshot_name = optarg;

            break;

        case 'S':

        {

            int64_t sval;

            char *end;

            sval = strtosz_suffix(optarg, &end, STRTOSZ_DEFSUFFIX_B);

            if (sval < 0 || *end) {

                error_report("Invalid minimum zero buffer size for sparse output specified");

                return 1;

            }



            min_sparse = sval / BDRV_SECTOR_SIZE;

            break;

        }

        case 'p':

            progress = 1;

            break;

        case 't':

            cache = optarg;

            break;

        }

    }



    bs_n = argc - optind - 1;

    if (bs_n < 1) {

        help();

    }



    out_filename = argv[argc - 1];



    /* Initialize before goto out */

    qemu_progress_init(progress, 2.0);



    if (options && !strcmp(options, "?")) {

        ret = print_block_option_help(out_filename, out_fmt);

        goto out;

    }



    if (bs_n > 1 && out_baseimg) {

        error_report("-B makes no sense when concatenating multiple input "

                     "images");

        ret = -1;

        goto out;

    }



    qemu_progress_print(0, 100);



    bs = g_malloc0(bs_n * sizeof(BlockDriverState *));



    total_sectors = 0;

    for (bs_i = 0; bs_i < bs_n; bs_i++) {

        bs[bs_i] = bdrv_new_open(argv[optind + bs_i], fmt, BDRV_O_FLAGS);

        if (!bs[bs_i]) {

            error_report("Could not open '%s'", argv[optind + bs_i]);

            ret = -1;

            goto out;

        }

        bdrv_get_geometry(bs[bs_i], &bs_sectors);

        total_sectors += bs_sectors;

    }



    if (snapshot_name != NULL) {

        if (bs_n > 1) {

            error_report("No support for concatenating multiple snapshot");

            ret = -1;

            goto out;

        }

        if (bdrv_snapshot_load_tmp(bs[0], snapshot_name) < 0) {

            error_report("Failed to load snapshot");

            ret = -1;

            goto out;

        }

    }



    /* Find driver and parse its options */

    drv = bdrv_find_format(out_fmt);

    if (!drv) {

        error_report("Unknown file format '%s'", out_fmt);

        ret = -1;

        goto out;

    }



    proto_drv = bdrv_find_protocol(out_filename);

    if (!proto_drv) {

        error_report("Unknown protocol '%s'", out_filename);

        ret = -1;

        goto out;

    }



    create_options = append_option_parameters(create_options,

                                              drv->create_options);

    create_options = append_option_parameters(create_options,

                                              proto_drv->create_options);



    if (options) {

        param = parse_option_parameters(options, create_options, param);

        if (param == NULL) {

            error_report("Invalid options for file format '%s'.", out_fmt);

            ret = -1;

            goto out;

        }

    } else {

        param = parse_option_parameters("", create_options, param);

    }



    set_option_parameter_int(param, BLOCK_OPT_SIZE, total_sectors * 512);

    ret = add_old_style_options(out_fmt, param, out_baseimg, NULL);

    if (ret < 0) {

        goto out;

    }



    /* Get backing file name if -o backing_file was used */

    out_baseimg_param = get_option_parameter(param, BLOCK_OPT_BACKING_FILE);

    if (out_baseimg_param) {

        out_baseimg = out_baseimg_param->value.s;

    }



    /* Check if compression is supported */

    if (compress) {

        QEMUOptionParameter *encryption =

            get_option_parameter(param, BLOCK_OPT_ENCRYPT);

        QEMUOptionParameter *preallocation =

            get_option_parameter(param, BLOCK_OPT_PREALLOC);



        if (!drv->bdrv_write_compressed) {

            error_report("Compression not supported for this file format");

            ret = -1;

            goto out;

        }



        if (encryption && encryption->value.n) {

            error_report("Compression and encryption not supported at "

                         "the same time");

            ret = -1;

            goto out;

        }



        if (preallocation && preallocation->value.s

            && strcmp(preallocation->value.s, "off"))

        {

            error_report("Compression and preallocation not supported at "

                         "the same time");

            ret = -1;

            goto out;

        }

    }



    /* Create the new image */

    ret = bdrv_create(drv, out_filename, param);

    if (ret < 0) {

        if (ret == -ENOTSUP) {

            error_report("Formatting not supported for file format '%s'",

                         out_fmt);

        } else if (ret == -EFBIG) {

            error_report("The image size is too large for file format '%s'",

                         out_fmt);

        } else {

            error_report("%s: error while converting %s: %s",

                         out_filename, out_fmt, strerror(-ret));

        }

        goto out;

    }



    flags = BDRV_O_RDWR;

    ret = bdrv_parse_cache_flags(cache, &flags);

    if (ret < 0) {

        error_report("Invalid cache option: %s", cache);

        return -1;

    }



    out_bs = bdrv_new_open(out_filename, out_fmt, flags);

    if (!out_bs) {

        ret = -1;

        goto out;

    }



    bs_i = 0;

    bs_offset = 0;

    bdrv_get_geometry(bs[0], &bs_sectors);

    buf = qemu_blockalign(out_bs, IO_BUF_SIZE);



    if (compress) {

        ret = bdrv_get_info(out_bs, &bdi);

        if (ret < 0) {

            error_report("could not get block driver info");

            goto out;

        }

        cluster_size = bdi.cluster_size;

        if (cluster_size <= 0 || cluster_size > IO_BUF_SIZE) {

            error_report("invalid cluster size");

            ret = -1;

            goto out;

        }

        cluster_sectors = cluster_size >> 9;

        sector_num = 0;



        nb_sectors = total_sectors;

        local_progress = (float)100 /

            (nb_sectors / MIN(nb_sectors, cluster_sectors));



        for(;;) {

            int64_t bs_num;

            int remainder;

            uint8_t *buf2;



            nb_sectors = total_sectors - sector_num;

            if (nb_sectors <= 0)

                break;

            if (nb_sectors >= cluster_sectors)

                n = cluster_sectors;

            else

                n = nb_sectors;



            bs_num = sector_num - bs_offset;

            assert (bs_num >= 0);

            remainder = n;

            buf2 = buf;

            while (remainder > 0) {

                int nlow;

                while (bs_num == bs_sectors) {

                    bs_i++;

                    assert (bs_i < bs_n);

                    bs_offset += bs_sectors;

                    bdrv_get_geometry(bs[bs_i], &bs_sectors);

                    bs_num = 0;

                    /* printf("changing part: sector_num=%" PRId64 ", "

                       "bs_i=%d, bs_offset=%" PRId64 ", bs_sectors=%" PRId64

                       "\n", sector_num, bs_i, bs_offset, bs_sectors); */

                }

                assert (bs_num < bs_sectors);



                nlow = (remainder > bs_sectors - bs_num) ? bs_sectors - bs_num : remainder;



                ret = bdrv_read(bs[bs_i], bs_num, buf2, nlow);

                if (ret < 0) {

                    error_report("error while reading sector %" PRId64 ": %s",

                                 bs_num, strerror(-ret));

                    goto out;

                }



                buf2 += nlow * 512;

                bs_num += nlow;



                remainder -= nlow;

            }

            assert (remainder == 0);



            if (n < cluster_sectors) {

                memset(buf + n * 512, 0, cluster_size - n * 512);

            }

            if (!buffer_is_zero(buf, cluster_size)) {

                ret = bdrv_write_compressed(out_bs, sector_num, buf,

                                            cluster_sectors);

                if (ret != 0) {

                    error_report("error while compressing sector %" PRId64

                                 ": %s", sector_num, strerror(-ret));

                    goto out;

                }

            }

            sector_num += n;

            qemu_progress_print(local_progress, 100);

        }

        /* signal EOF to align */

        bdrv_write_compressed(out_bs, 0, NULL, 0);

    } else {

        int has_zero_init = bdrv_has_zero_init(out_bs);



        sector_num = 0; // total number of sectors converted so far

        nb_sectors = total_sectors - sector_num;

        local_progress = (float)100 /

            (nb_sectors / MIN(nb_sectors, IO_BUF_SIZE / 512));



        for(;;) {

            nb_sectors = total_sectors - sector_num;

            if (nb_sectors <= 0) {

                break;

            }

            if (nb_sectors >= (IO_BUF_SIZE / 512)) {

                n = (IO_BUF_SIZE / 512);

            } else {

                n = nb_sectors;

            }



            while (sector_num - bs_offset >= bs_sectors) {

                bs_i ++;

                assert (bs_i < bs_n);

                bs_offset += bs_sectors;

                bdrv_get_geometry(bs[bs_i], &bs_sectors);

                /* printf("changing part: sector_num=%" PRId64 ", bs_i=%d, "

                  "bs_offset=%" PRId64 ", bs_sectors=%" PRId64 "\n",

                   sector_num, bs_i, bs_offset, bs_sectors); */

            }



            if (n > bs_offset + bs_sectors - sector_num) {

                n = bs_offset + bs_sectors - sector_num;

            }



            if (has_zero_init) {

                /* If the output image is being created as a copy on write image,

                   assume that sectors which are unallocated in the input image

                   are present in both the output's and input's base images (no

                   need to copy them). */

                if (out_baseimg) {

                    if (!bdrv_is_allocated(bs[bs_i], sector_num - bs_offset,

                                           n, &n1)) {

                        sector_num += n1;

                        continue;

                    }

                    /* The next 'n1' sectors are allocated in the input image. Copy

                       only those as they may be followed by unallocated sectors. */

                    n = n1;

                }

            } else {

                n1 = n;

            }



            ret = bdrv_read(bs[bs_i], sector_num - bs_offset, buf, n);

            if (ret < 0) {

                error_report("error while reading sector %" PRId64 ": %s",

                             sector_num - bs_offset, strerror(-ret));

                goto out;

            }

            /* NOTE: at the same time we convert, we do not write zero

               sectors to have a chance to compress the image. Ideally, we

               should add a specific call to have the info to go faster */

            buf1 = buf;

            while (n > 0) {

                /* If the output image is being created as a copy on write image,

                   copy all sectors even the ones containing only NUL bytes,

                   because they may differ from the sectors in the base image.



                   If the output is to a host device, we also write out

                   sectors that are entirely 0, since whatever data was

                   already there is garbage, not 0s. */

                if (!has_zero_init || out_baseimg ||

                    is_allocated_sectors_min(buf1, n, &n1, min_sparse)) {

                    ret = bdrv_write(out_bs, sector_num, buf1, n1);

                    if (ret < 0) {

                        error_report("error while writing sector %" PRId64

                                     ": %s", sector_num, strerror(-ret));

                        goto out;

                    }

                }

                sector_num += n1;

                n -= n1;

                buf1 += n1 * 512;

            }

            qemu_progress_print(local_progress, 100);

        }

    }

out:

    qemu_progress_end();

    free_option_parameters(create_options);

    free_option_parameters(param);

    qemu_vfree(buf);

    if (out_bs) {

        bdrv_delete(out_bs);

    }

    if (bs) {

        for (bs_i = 0; bs_i < bs_n; bs_i++) {

            if (bs[bs_i]) {

                bdrv_delete(bs[bs_i]);

            }

        }

        g_free(bs);

    }

    if (ret) {

        return 1;

    }

    return 0;

}
