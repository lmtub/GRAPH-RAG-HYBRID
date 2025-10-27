static int iscsi_open(BlockDriverState *bs, QDict *options, int flags,
                      Error **errp)
{
    IscsiLun *iscsilun = bs->opaque;
    struct iscsi_context *iscsi = NULL;
    struct iscsi_url *iscsi_url = NULL;
    struct scsi_task *task = NULL;
    struct scsi_inquiry_standard *inq = NULL;
    struct scsi_inquiry_supported_pages *inq_vpd;
    char *initiator_name = NULL;
    QemuOpts *opts;
    Error *local_err = NULL;
    const char *filename;
    int i, ret;
    if ((BDRV_SECTOR_SIZE % 512) != 0) {
        error_setg(errp, "iSCSI: Invalid BDRV_SECTOR_SIZE. "
                   "BDRV_SECTOR_SIZE(%lld) is not a multiple "
                   "of 512", BDRV_SECTOR_SIZE);
        return -EINVAL;
    opts = qemu_opts_create(&runtime_opts, NULL, 0, &error_abort);
    qemu_opts_absorb_qdict(opts, options, &local_err);
    if (local_err) {
        error_propagate(errp, local_err);
        ret = -EINVAL;
    filename = qemu_opt_get(opts, "filename");
    iscsi_url = iscsi_parse_full_url(iscsi, filename);
    if (iscsi_url == NULL) {
        error_setg(errp, "Failed to parse URL : %s", filename);
        ret = -EINVAL;
    memset(iscsilun, 0, sizeof(IscsiLun));
    initiator_name = parse_initiator_name(iscsi_url->target);
    iscsi = iscsi_create_context(initiator_name);
    if (iscsi == NULL) {
        error_setg(errp, "iSCSI: Failed to create iSCSI context.");
        ret = -ENOMEM;
    if (iscsi_set_targetname(iscsi, iscsi_url->target)) {
        error_setg(errp, "iSCSI: Failed to set target name.");
        ret = -EINVAL;
    if (iscsi_url->user != NULL) {
        ret = iscsi_set_initiator_username_pwd(iscsi, iscsi_url->user,
                                              iscsi_url->passwd);
        if (ret != 0) {
            error_setg(errp, "Failed to set initiator username and password");
            ret = -EINVAL;
    /* check if we got CHAP username/password via the options */
    parse_chap(iscsi, iscsi_url->target, &local_err);
    if (local_err != NULL) {
        error_propagate(errp, local_err);
        ret = -EINVAL;
    if (iscsi_set_session_type(iscsi, ISCSI_SESSION_NORMAL) != 0) {
        error_setg(errp, "iSCSI: Failed to set session type to normal.");
        ret = -EINVAL;
    iscsi_set_header_digest(iscsi, ISCSI_HEADER_DIGEST_NONE_CRC32C);
    /* check if we got HEADER_DIGEST via the options */
    parse_header_digest(iscsi, iscsi_url->target, &local_err);
    if (local_err != NULL) {
        error_propagate(errp, local_err);
        ret = -EINVAL;
    if (iscsi_full_connect_sync(iscsi, iscsi_url->portal, iscsi_url->lun) != 0) {
        error_setg(errp, "iSCSI: Failed to connect to LUN : %s",
            iscsi_get_error(iscsi));
        ret = -EINVAL;
    iscsilun->iscsi = iscsi;
    iscsilun->aio_context = bdrv_get_aio_context(bs);
    iscsilun->lun   = iscsi_url->lun;
    iscsilun->has_write_same = true;
    task = iscsi_do_inquiry(iscsilun->iscsi, iscsilun->lun, 0, 0,
                            (void **) &inq, errp);
    if (task == NULL) {
        ret = -EINVAL;
    iscsilun->type = inq->periperal_device_type;
    scsi_free_scsi_task(task);
    task = NULL;
    iscsi_readcapacity_sync(iscsilun, &local_err);
    if (local_err != NULL) {
        error_propagate(errp, local_err);
        ret = -EINVAL;
    bs->total_sectors = sector_lun2qemu(iscsilun->num_blocks, iscsilun);
    bs->request_alignment = iscsilun->block_size;
    /* We don't have any emulation for devices other than disks and CD-ROMs, so
     * this must be sg ioctl compatible. We force it to be sg, otherwise qemu
     * will try to read from the device to guess the image format.
     */
    if (iscsilun->type != TYPE_DISK && iscsilun->type != TYPE_ROM) {
        bs->sg = 1;
    task = iscsi_do_inquiry(iscsilun->iscsi, iscsilun->lun, 1,
                            SCSI_INQUIRY_PAGECODE_SUPPORTED_VPD_PAGES,
                            (void **) &inq_vpd, errp);
    if (task == NULL) {
        ret = -EINVAL;
    for (i = 0; i < inq_vpd->num_pages; i++) {
        struct scsi_task *inq_task;
        struct scsi_inquiry_logical_block_provisioning *inq_lbp;
        struct scsi_inquiry_block_limits *inq_bl;
        switch (inq_vpd->pages[i]) {
        case SCSI_INQUIRY_PAGECODE_LOGICAL_BLOCK_PROVISIONING:
            inq_task = iscsi_do_inquiry(iscsilun->iscsi, iscsilun->lun, 1,
                                        SCSI_INQUIRY_PAGECODE_LOGICAL_BLOCK_PROVISIONING,
                                        (void **) &inq_lbp, errp);
            if (inq_task == NULL) {
                ret = -EINVAL;
            memcpy(&iscsilun->lbp, inq_lbp,
                   sizeof(struct scsi_inquiry_logical_block_provisioning));
            scsi_free_scsi_task(inq_task);
            break;
        case SCSI_INQUIRY_PAGECODE_BLOCK_LIMITS:
            inq_task = iscsi_do_inquiry(iscsilun->iscsi, iscsilun->lun, 1,
                                    SCSI_INQUIRY_PAGECODE_BLOCK_LIMITS,
                                    (void **) &inq_bl, errp);
            if (inq_task == NULL) {
                ret = -EINVAL;
            memcpy(&iscsilun->bl, inq_bl,
                   sizeof(struct scsi_inquiry_block_limits));
            scsi_free_scsi_task(inq_task);
            break;
        default:
            break;
    scsi_free_scsi_task(task);
    task = NULL;
    iscsi_attach_aio_context(bs, iscsilun->aio_context);
    /* Guess the internal cluster (page) size of the iscsi target by the means
     * of opt_unmap_gran. Transfer the unmap granularity only if it has a
     * reasonable size */
    if (iscsilun->bl.opt_unmap_gran * iscsilun->block_size >= 4 * 1024 &&
        iscsilun->bl.opt_unmap_gran * iscsilun->block_size <= 16 * 1024 * 1024) {
        iscsilun->cluster_sectors = (iscsilun->bl.opt_unmap_gran *
                                     iscsilun->block_size) >> BDRV_SECTOR_BITS;
        if (iscsilun->lbprz && !(bs->open_flags & BDRV_O_NOCACHE)) {
            iscsilun->allocationmap = iscsi_allocationmap_init(iscsilun);
            if (iscsilun->allocationmap == NULL) {
                ret = -ENOMEM;
out:
    qemu_opts_del(opts);
    g_free(initiator_name);
    if (iscsi_url != NULL) {
        iscsi_destroy_url(iscsi_url);
    if (task != NULL) {
        scsi_free_scsi_task(task);
    if (ret) {
        if (iscsi != NULL) {
            iscsi_destroy_context(iscsi);
        memset(iscsilun, 0, sizeof(IscsiLun));
    return ret;