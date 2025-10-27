static int mov_read_ftyp(MOVContext *c, AVIOContext *pb, MOVAtom atom)

{

    uint32_t minor_ver;

    int comp_brand_size;

    char minor_ver_str[11]; /* 32 bit integer -> 10 digits + null */

    char* comp_brands_str;

    uint8_t type[5] = {0};



    avio_read(pb, type, 4);

    if (strcmp(type, "qt  "))

        c->isom = 1;

    av_log(c->fc, AV_LOG_DEBUG, "ISO: File Type Major Brand: %.4s\n",(char *)&type);

    av_dict_set(&c->fc->metadata, "major_brand", type, 0);

    minor_ver = avio_rb32(pb); /* minor version */

    snprintf(minor_ver_str, sizeof(minor_ver_str), "%"PRIu32"", minor_ver);

    av_dict_set(&c->fc->metadata, "minor_version", minor_ver_str, 0);



    comp_brand_size = atom.size - 8;

    if (comp_brand_size < 0)

        return AVERROR_INVALIDDATA;

    comp_brands_str = av_malloc(comp_brand_size + 1); /* Add null terminator */

    if (!comp_brands_str)

        return AVERROR(ENOMEM);

    avio_read(pb, comp_brands_str, comp_brand_size);

    comp_brands_str[comp_brand_size] = 0;

    av_dict_set(&c->fc->metadata, "compatible_brands", comp_brands_str, 0);

    av_freep(&comp_brands_str);



    return 0;

}
