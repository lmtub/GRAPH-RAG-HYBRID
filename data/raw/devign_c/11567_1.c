static int vpc_has_zero_init(BlockDriverState *bs)

{

    BDRVVPCState *s = bs->opaque;

    VHDFooter *footer =  (VHDFooter *) s->footer_buf;



    if (cpu_to_be32(footer->type) == VHD_FIXED) {

        return bdrv_has_zero_init(bs->file);

    } else {

        return 1;

    }

}
