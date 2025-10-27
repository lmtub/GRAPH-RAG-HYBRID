static void create_header32(DumpState *s, Error **errp)
{
    DiskDumpHeader32 *dh = NULL;
    KdumpSubHeader32 *kh = NULL;
    size_t size;
    uint32_t block_size;
    uint32_t sub_hdr_size;
    uint32_t bitmap_blocks;
    uint32_t status = 0;
    uint64_t offset_note;
    Error *local_err = NULL;
    /* write common header, the version of kdump-compressed format is 6th */
    size = sizeof(DiskDumpHeader32);
    dh = g_malloc0(size);
    strncpy(dh->signature, KDUMP_SIGNATURE, strlen(KDUMP_SIGNATURE));
    dh->header_version = cpu_to_dump32(s, 6);
    block_size = s->dump_info.page_size;
    dh->block_size = cpu_to_dump32(s, block_size);
    sub_hdr_size = sizeof(struct KdumpSubHeader32) + s->note_size;
    sub_hdr_size = DIV_ROUND_UP(sub_hdr_size, block_size);
    dh->sub_hdr_size = cpu_to_dump32(s, sub_hdr_size);
    /* dh->max_mapnr may be truncated, full 64bit is in kh.max_mapnr_64 */
    dh->max_mapnr = cpu_to_dump32(s, MIN(s->max_mapnr, UINT_MAX));
    dh->nr_cpus = cpu_to_dump32(s, s->nr_cpus);
    bitmap_blocks = DIV_ROUND_UP(s->len_dump_bitmap, block_size) * 2;
    dh->bitmap_blocks = cpu_to_dump32(s, bitmap_blocks);
    strncpy(dh->utsname.machine, ELF_MACHINE_UNAME, sizeof(dh->utsname.machine));
    if (s->flag_compress & DUMP_DH_COMPRESSED_ZLIB) {
        status |= DUMP_DH_COMPRESSED_ZLIB;
#ifdef CONFIG_LZO
    if (s->flag_compress & DUMP_DH_COMPRESSED_LZO) {
        status |= DUMP_DH_COMPRESSED_LZO;
#endif
#ifdef CONFIG_SNAPPY
    if (s->flag_compress & DUMP_DH_COMPRESSED_SNAPPY) {
        status |= DUMP_DH_COMPRESSED_SNAPPY;
#endif
    dh->status = cpu_to_dump32(s, status);
    if (write_buffer(s->fd, 0, dh, size) < 0) {
        error_setg(errp, "dump: failed to write disk dump header");
        goto out;
    /* write sub header */
    size = sizeof(KdumpSubHeader32);
    kh = g_malloc0(size);
    /* 64bit max_mapnr_64 */
    kh->max_mapnr_64 = cpu_to_dump64(s, s->max_mapnr);
    kh->phys_base = cpu_to_dump32(s, s->dump_info.phys_base);
    kh->dump_level = cpu_to_dump32(s, DUMP_LEVEL);
    offset_note = DISKDUMP_HEADER_BLOCKS * block_size + size;
    kh->offset_note = cpu_to_dump64(s, offset_note);
    kh->note_size = cpu_to_dump32(s, s->note_size);
    if (write_buffer(s->fd, DISKDUMP_HEADER_BLOCKS *
                     block_size, kh, size) < 0) {
        error_setg(errp, "dump: failed to write kdump sub header");
        goto out;
    /* write note */
    s->note_buf = g_malloc0(s->note_size);
    s->note_buf_offset = 0;
    /* use s->note_buf to store notes temporarily */
    write_elf32_notes(buf_write_note, s, &local_err);
    if (local_err) {
        error_propagate(errp, local_err);
        goto out;
    if (write_buffer(s->fd, offset_note, s->note_buf,
                     s->note_size) < 0) {
        error_setg(errp, "dump: failed to write notes");
        goto out;
    /* get offset of dump_bitmap */
    s->offset_dump_bitmap = (DISKDUMP_HEADER_BLOCKS + sub_hdr_size) *
                             block_size;
    /* get offset of page */
    s->offset_page = (DISKDUMP_HEADER_BLOCKS + sub_hdr_size + bitmap_blocks) *
                     block_size;
out:
    g_free(dh);
    g_free(kh);
    g_free(s->note_buf);