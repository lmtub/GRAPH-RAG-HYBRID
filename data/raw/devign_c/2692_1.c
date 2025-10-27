static int vmdk_snapshot_create(const char *filename, const char *backing_file)

{

    int snp_fd, p_fd;

    int ret;

    uint32_t p_cid;

    char *p_name, *gd_buf, *rgd_buf;

    const char *real_filename, *temp_str;

    VMDK4Header header;

    uint32_t gde_entries, gd_size;

    int64_t gd_offset, rgd_offset, capacity, gt_size;

    char p_desc[DESC_SIZE], s_desc[DESC_SIZE], hdr[HEADER_SIZE];

    static const char desc_template[] =

    "# Disk DescriptorFile\n"

    "version=1\n"

    "CID=%x\n"

    "parentCID=%x\n"

    "createType=\"monolithicSparse\"\n"

    "parentFileNameHint=\"%s\"\n"

    "\n"

    "# Extent description\n"

    "RW %u SPARSE \"%s\"\n"

    "\n"

    "# The Disk Data Base \n"

    "#DDB\n"

    "\n";



    snp_fd = open(filename, O_RDWR | O_CREAT | O_TRUNC | O_BINARY | O_LARGEFILE, 0644);

    if (snp_fd < 0)

        return -errno;

    p_fd = open(backing_file, O_RDONLY | O_BINARY | O_LARGEFILE);

    if (p_fd < 0) {

        close(snp_fd);

        return -errno;

    }



    /* read the header */

    if (lseek(p_fd, 0x0, SEEK_SET) == -1) {

        ret = -errno;

        goto fail;

    }

    if (read(p_fd, hdr, HEADER_SIZE) != HEADER_SIZE) {

        ret = -errno;

        goto fail;

    }



    /* write the header */

    if (lseek(snp_fd, 0x0, SEEK_SET) == -1) {

        ret = -errno;

        goto fail;

    }

    if (write(snp_fd, hdr, HEADER_SIZE) == -1) {

        ret = -errno;

        goto fail;

    }



    memset(&header, 0, sizeof(header));

    memcpy(&header,&hdr[4], sizeof(header)); // skip the VMDK4_MAGIC



    if (ftruncate(snp_fd, header.grain_offset << 9)) {

        ret = -errno;

        goto fail;

    }

    /* the descriptor offset = 0x200 */

    if (lseek(p_fd, 0x200, SEEK_SET) == -1) {

        ret = -errno;

        goto fail;

    }

    if (read(p_fd, p_desc, DESC_SIZE) != DESC_SIZE) {

        ret = -errno;

        goto fail;

    }



    if ((p_name = strstr(p_desc,"CID")) != NULL) {

        p_name += sizeof("CID");

        sscanf(p_name,"%x",&p_cid);

    }



    real_filename = filename;

    if ((temp_str = strrchr(real_filename, '\\')) != NULL)

        real_filename = temp_str + 1;

    if ((temp_str = strrchr(real_filename, '/')) != NULL)

        real_filename = temp_str + 1;

    if ((temp_str = strrchr(real_filename, ':')) != NULL)

        real_filename = temp_str + 1;



    snprintf(s_desc, sizeof(s_desc), desc_template, p_cid, p_cid, backing_file,

             (uint32_t)header.capacity, real_filename);



    /* write the descriptor */

    if (lseek(snp_fd, 0x200, SEEK_SET) == -1) {

        ret = -errno;

        goto fail;

    }

    if (write(snp_fd, s_desc, strlen(s_desc)) == -1) {

        ret = -errno;

        goto fail;

    }



    gd_offset = header.gd_offset * SECTOR_SIZE;     // offset of GD table

    rgd_offset = header.rgd_offset * SECTOR_SIZE;   // offset of RGD table

    capacity = header.capacity * SECTOR_SIZE;       // Extent size

    /*

     * Each GDE span 32M disk, means:

     * 512 GTE per GT, each GTE points to grain

     */

    gt_size = (int64_t)header.num_gtes_per_gte * header.granularity * SECTOR_SIZE;

    if (!gt_size) {

        ret = -EINVAL;

        goto fail;

    }

    gde_entries = (uint32_t)(capacity / gt_size);  // number of gde/rgde

    gd_size = gde_entries * sizeof(uint32_t);



    /* write RGD */

    rgd_buf = qemu_malloc(gd_size);

    if (lseek(p_fd, rgd_offset, SEEK_SET) == -1) {

        ret = -errno;

        goto fail_rgd;

    }

    if (read(p_fd, rgd_buf, gd_size) != gd_size) {

        ret = -errno;

        goto fail_rgd;

    }

    if (lseek(snp_fd, rgd_offset, SEEK_SET) == -1) {

        ret = -errno;

        goto fail_rgd;

    }

    if (write(snp_fd, rgd_buf, gd_size) == -1) {

        ret = -errno;

        goto fail_rgd;

    }

    qemu_free(rgd_buf);



    /* write GD */

    gd_buf = qemu_malloc(gd_size);

    if (lseek(p_fd, gd_offset, SEEK_SET) == -1) {

        ret = -errno;

        goto fail_gd;

    }

    if (read(p_fd, gd_buf, gd_size) != gd_size) {

        ret = -errno;

        goto fail_gd;

    }

    if (lseek(snp_fd, gd_offset, SEEK_SET) == -1) {

        ret = -errno;

        goto fail_gd;

    }

    if (write(snp_fd, gd_buf, gd_size) == -1) {

        ret = -errno;

        goto fail_gd;

    }

    qemu_free(gd_buf);



    close(p_fd);

    close(snp_fd);

    return 0;



    fail_gd:

    qemu_free(gd_buf);

    fail_rgd:

    qemu_free(rgd_buf);

    fail:

    close(p_fd);

    close(snp_fd);

    return ret;

}
