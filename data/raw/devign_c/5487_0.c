static void sd_reset(SDState *sd, BlockBackend *blk)

{

    uint64_t size;

    uint64_t sect;



    if (blk) {

        blk_get_geometry(blk, &sect);

    } else {

        sect = 0;

    }

    size = sect << 9;



    sect = sd_addr_to_wpnum(size) + 1;



    sd->state = sd_idle_state;

    sd->rca = 0x0000;

    sd_set_ocr(sd);

    sd_set_scr(sd);

    sd_set_cid(sd);

    sd_set_csd(sd, size);

    sd_set_cardstatus(sd);

    sd_set_sdstatus(sd);



    sd->blk = blk;



    if (sd->wp_groups)

        g_free(sd->wp_groups);

    sd->wp_switch = blk ? blk_is_read_only(blk) : false;

    sd->wpgrps_size = sect;

    sd->wp_groups = bitmap_new(sd->wpgrps_size);

    memset(sd->function_group, 0, sizeof(sd->function_group));

    sd->erase_start = 0;

    sd->erase_end = 0;

    sd->size = size;

    sd->blk_len = 0x200;

    sd->pwd_len = 0;

    sd->expecting_acmd = false;

}
