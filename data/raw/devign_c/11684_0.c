static int vpc_create(const char *filename, QemuOpts *opts, Error **errp)

{

    uint8_t buf[1024];

    VHDFooter *footer = (VHDFooter *) buf;

    char *disk_type_param;

    int fd, i;

    uint16_t cyls = 0;

    uint8_t heads = 0;

    uint8_t secs_per_cyl = 0;

    int64_t total_sectors;

    int64_t total_size;

    int disk_type;

    int ret = -EIO;

    bool nocow = false;



    /* Read out options */

    total_size = qemu_opt_get_size_del(opts, BLOCK_OPT_SIZE, 0);

    disk_type_param = qemu_opt_get_del(opts, BLOCK_OPT_SUBFMT);

    if (disk_type_param) {

        if (!strcmp(disk_type_param, "dynamic")) {

            disk_type = VHD_DYNAMIC;

        } else if (!strcmp(disk_type_param, "fixed")) {

            disk_type = VHD_FIXED;

        } else {

            ret = -EINVAL;

            goto out;

        }

    } else {

        disk_type = VHD_DYNAMIC;

    }

    nocow = qemu_opt_get_bool_del(opts, BLOCK_OPT_NOCOW, false);



    /* Create the file */

    fd = qemu_open(filename, O_WRONLY | O_CREAT | O_TRUNC | O_BINARY, 0644);

    if (fd < 0) {

        ret = -EIO;

        goto out;

    }



    if (nocow) {

#ifdef __linux__

        /* Set NOCOW flag to solve performance issue on fs like btrfs.

         * This is an optimisation. The FS_IOC_SETFLAGS ioctl return value will

         * be ignored since any failure of this operation should not block the

         * left work.

         */

        int attr;

        if (ioctl(fd, FS_IOC_GETFLAGS, &attr) == 0) {

            attr |= FS_NOCOW_FL;

            ioctl(fd, FS_IOC_SETFLAGS, &attr);

        }

#endif

    }



    /*

     * Calculate matching total_size and geometry. Increase the number of

     * sectors requested until we get enough (or fail). This ensures that

     * qemu-img convert doesn't truncate images, but rather rounds up.

     */

    total_sectors = total_size / BDRV_SECTOR_SIZE;

    for (i = 0; total_sectors > (int64_t)cyls * heads * secs_per_cyl; i++) {

        if (calculate_geometry(total_sectors + i, &cyls, &heads,

                               &secs_per_cyl))

        {

            ret = -EFBIG;

            goto fail;

        }

    }



    total_sectors = (int64_t) cyls * heads * secs_per_cyl;



    /* Prepare the Hard Disk Footer */

    memset(buf, 0, 1024);



    memcpy(footer->creator, "conectix", 8);

    /* TODO Check if "qemu" creator_app is ok for VPC */

    memcpy(footer->creator_app, "qemu", 4);

    memcpy(footer->creator_os, "Wi2k", 4);



    footer->features = be32_to_cpu(0x02);

    footer->version = be32_to_cpu(0x00010000);

    if (disk_type == VHD_DYNAMIC) {

        footer->data_offset = be64_to_cpu(HEADER_SIZE);

    } else {

        footer->data_offset = be64_to_cpu(0xFFFFFFFFFFFFFFFFULL);

    }

    footer->timestamp = be32_to_cpu(time(NULL) - VHD_TIMESTAMP_BASE);



    /* Version of Virtual PC 2007 */

    footer->major = be16_to_cpu(0x0005);

    footer->minor = be16_to_cpu(0x0003);

    if (disk_type == VHD_DYNAMIC) {

        footer->orig_size = be64_to_cpu(total_sectors * 512);

        footer->size = be64_to_cpu(total_sectors * 512);

    } else {

        footer->orig_size = be64_to_cpu(total_size);

        footer->size = be64_to_cpu(total_size);

    }

    footer->cyls = be16_to_cpu(cyls);

    footer->heads = heads;

    footer->secs_per_cyl = secs_per_cyl;



    footer->type = be32_to_cpu(disk_type);



#if defined(CONFIG_UUID)

    uuid_generate(footer->uuid);

#endif



    footer->checksum = be32_to_cpu(vpc_checksum(buf, HEADER_SIZE));



    if (disk_type == VHD_DYNAMIC) {

        ret = create_dynamic_disk(fd, buf, total_sectors);

    } else {

        ret = create_fixed_disk(fd, buf, total_size);

    }



fail:

    qemu_close(fd);

out:

    g_free(disk_type_param);

    return ret;

}
