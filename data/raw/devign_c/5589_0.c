static void end_ebml_master_crc32(AVIOContext *pb, AVIOContext **dyn_cp, MatroskaMuxContext *mkv,

                                  ebml_master master)

{

    uint8_t *buf, crc[4];

    int size;



    if (pb->seekable) {

        size = avio_close_dyn_buf(*dyn_cp, &buf);

        if (mkv->write_crc && mkv->mode != MODE_WEBM) {

            AV_WL32(crc, av_crc(av_crc_get_table(AV_CRC_32_IEEE_LE), UINT32_MAX, buf, size) ^ UINT32_MAX);

            put_ebml_binary(pb, EBML_ID_CRC32, crc, sizeof(crc));

        }

        avio_write(pb, buf, size);

        end_ebml_master(pb, master);

    } else {

        end_ebml_master(*dyn_cp, master);

        size = avio_close_dyn_buf(*dyn_cp, &buf);

        avio_write(pb, buf, size);

    }

    av_free(buf);

    *dyn_cp = NULL;

}
