static int vmdk_parse_extents(const char *desc, BlockDriverState *bs,

                              const char *desc_file_path, Error **errp)

{

    int ret;

    char access[11];

    char type[11];

    char fname[512];

    const char *p = desc;

    int64_t sectors = 0;

    int64_t flat_offset;

    char extent_path[PATH_MAX];

    BlockDriverState *extent_file;

    BDRVVmdkState *s = bs->opaque;

    VmdkExtent *extent;



    while (*p) {

        /* parse extent line:

         * RW [size in sectors] FLAT "file-name.vmdk" OFFSET

         * or

         * RW [size in sectors] SPARSE "file-name.vmdk"

         */

        flat_offset = -1;

        ret = sscanf(p, "%10s %" SCNd64 " %10s \"%511[^\n\r\"]\" %" SCNd64,

                access, &sectors, type, fname, &flat_offset);

        if (ret < 4 || strcmp(access, "RW")) {

            goto next_line;

        } else if (!strcmp(type, "FLAT")) {

            if (ret != 5 || flat_offset < 0) {

                error_setg(errp, "Invalid extent lines: \n%s", p);

                return -EINVAL;

            }

        } else if (!strcmp(type, "VMFS")) {

            if (ret == 4) {

                flat_offset = 0;

            } else {

                error_setg(errp, "Invalid extent lines:\n%s", p);

                return -EINVAL;

            }

        } else if (ret != 4) {

            error_setg(errp, "Invalid extent lines:\n%s", p);

            return -EINVAL;

        }



        if (sectors <= 0 ||

            (strcmp(type, "FLAT") && strcmp(type, "SPARSE") &&

             strcmp(type, "VMFS") && strcmp(type, "VMFSSPARSE")) ||

            (strcmp(access, "RW"))) {

            goto next_line;

        }



        path_combine(extent_path, sizeof(extent_path),

                desc_file_path, fname);

        extent_file = NULL;

        ret = bdrv_open(&extent_file, extent_path, NULL, NULL,

                        bs->open_flags | BDRV_O_PROTOCOL, NULL, errp);

        if (ret) {

            return ret;

        }



        /* save to extents array */

        if (!strcmp(type, "FLAT") || !strcmp(type, "VMFS")) {

            /* FLAT extent */



            ret = vmdk_add_extent(bs, extent_file, true, sectors,

                            0, 0, 0, 0, 0, &extent, errp);

            if (ret < 0) {


                return ret;

            }

            extent->flat_start_offset = flat_offset << 9;

        } else if (!strcmp(type, "SPARSE") || !strcmp(type, "VMFSSPARSE")) {

            /* SPARSE extent and VMFSSPARSE extent are both "COWD" sparse file*/

            char *buf = vmdk_read_desc(extent_file, 0, errp);

            if (!buf) {

                ret = -EINVAL;

            } else {

                ret = vmdk_open_sparse(bs, extent_file, bs->open_flags, buf, errp);

            }

            if (ret) {

                g_free(buf);


                return ret;

            }

            extent = &s->extents[s->num_extents - 1];

        } else {

            error_setg(errp, "Unsupported extent type '%s'", type);


            return -ENOTSUP;

        }

        extent->type = g_strdup(type);

next_line:

        /* move to next line */

        while (*p) {

            if (*p == '\n') {

                p++;

                break;

            }

            p++;

        }

    }

    return 0;

}