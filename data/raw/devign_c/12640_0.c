static int mov_write_gmhd_tag(AVIOContext *pb)

{

    avio_wb32(pb, 0x20);   /* size */

    ffio_wfourcc(pb, "gmhd");

    avio_wb32(pb, 0x18);   /* gmin size */

    ffio_wfourcc(pb, "gmin");/* generic media info */

    avio_wb32(pb, 0);      /* version & flags */

    avio_wb16(pb, 0x40);   /* graphics mode = */

    avio_wb16(pb, 0x8000); /* opColor (r?) */

    avio_wb16(pb, 0x8000); /* opColor (g?) */

    avio_wb16(pb, 0x8000); /* opColor (b?) */

    avio_wb16(pb, 0);      /* balance */

    avio_wb16(pb, 0);      /* reserved */

    return 0x20;

}
