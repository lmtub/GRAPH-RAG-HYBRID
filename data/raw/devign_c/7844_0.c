static int create_header32(DumpState *s)

{

    int ret = 0;

    DiskDumpHeader32 *dh = NULL;

    KdumpSubHeader32 *kh = NULL;

    size_t size;

    int endian = s->dump_info.d_endian;

    uint32_t block_size;

    uint32_t sub_hdr_size;

    uint32_t bitmap_blocks;

    uint32_t status = 0;

    uint64_t offset_note;



    /* write common header, the version of kdump-compressed format is 6th */

    size = sizeof(DiskDumpHeader32);

    dh = g_malloc0(size);



    strncpy(dh->signature, KDUMP_SIGNATURE, strlen(KDUMP_SIGNATURE));

    dh->header_version = cpu_convert_to_target32(6, endian);

    block_size = s->page_size;

    dh->block_size = cpu_convert_to_target32(block_size, endian);

    sub_hdr_size = sizeof(struct KdumpSubHeader32) + s->note_size;

    sub_hdr_size = DIV_ROUND_UP(sub_hdr_size, block_size);

    dh->sub_hdr_size = cpu_convert_to_target32(sub_hdr_size, endian);

    /* dh->max_mapnr may be truncated, full 64bit is in kh.max_mapnr_64 */

    dh->max_mapnr = cpu_convert_to_target32(MIN(s->max_mapnr, UINT_MAX),

                                            endian);

    dh->nr_cpus = cpu_convert_to_target32(s->nr_cpus, endian);

    bitmap_blocks = DIV_ROUND_UP(s->len_dump_bitmap, block_size) * 2;

    dh->bitmap_blocks = cpu_convert_to_target32(bitmap_blocks, endian);

    strncpy(dh->utsname.machine, ELF_MACHINE_UNAME, sizeof(dh->utsname.machine));



    if (s->flag_compress & DUMP_DH_COMPRESSED_ZLIB) {

        status |= DUMP_DH_COMPRESSED_ZLIB;

    }

#ifdef CONFIG_LZO

    if (s->flag_compress & DUMP_DH_COMPRESSED_LZO) {

        status |= DUMP_DH_COMPRESSED_LZO;

    }

#endif

#ifdef CONFIG_SNAPPY

    if (s->flag_compress & DUMP_DH_COMPRESSED_SNAPPY) {

        status |= DUMP_DH_COMPRESSED_SNAPPY;

    }

#endif

    dh->status = cpu_convert_to_target32(status, endian);



    if (write_buffer(s->fd, 0, dh, size) < 0) {

        dump_error(s, "dump: failed to write disk dump header.\n");

        ret = -1;

        goto out;

    }



    /* write sub header */

    size = sizeof(KdumpSubHeader32);

    kh = g_malloc0(size);



    /* 64bit max_mapnr_64 */

    kh->max_mapnr_64 = cpu_convert_to_target64(s->max_mapnr, endian);

    kh->phys_base = cpu_convert_to_target32(PHYS_BASE, endian);

    kh->dump_level = cpu_convert_to_target32(DUMP_LEVEL, endian);



    offset_note = DISKDUMP_HEADER_BLOCKS * block_size + size;

    kh->offset_note = cpu_convert_to_target64(offset_note, endian);

    kh->note_size = cpu_convert_to_target32(s->note_size, endian);



    if (write_buffer(s->fd, DISKDUMP_HEADER_BLOCKS *

                     block_size, kh, size) < 0) {

        dump_error(s, "dump: failed to write kdump sub header.\n");

        ret = -1;

        goto out;

    }



    /* write note */

    s->note_buf = g_malloc0(s->note_size);

    s->note_buf_offset = 0;



    /* use s->note_buf to store notes temporarily */

    if (write_elf32_notes(buf_write_note, s) < 0) {

        ret = -1;

        goto out;

    }



    if (write_buffer(s->fd, offset_note, s->note_buf,

                     s->note_size) < 0) {

        dump_error(s, "dump: failed to write notes");

        ret = -1;

        goto out;

    }



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



    return ret;

}
